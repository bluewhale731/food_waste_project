import argparse
import json
import os
import re
import time

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

SEED = 42  # integer seed; the draft's "0.42" is a typo — seeds cannot be floats

# ---------------------------------------------------------------------------
# Scoring constants (Section II-E "Scoring Mechanics" of the paper)
# ---------------------------------------------------------------------------
# W_i: category-default unit weights in pounds, assigned because item weight
# cannot be inferred from a photograph alone.
CATEGORY_WEIGHTS_LB = {"produce": 1.0, "packaged": 1.5, "bakery": 1.0}

# V_i fallback: category-default nutrition value used when the OpenFoodFacts
# lookup fails (typical for generic, non-barcoded produce).
CATEGORY_DEFAULT_NUTRITION = {"produce": 0.85, "bakery": 0.60, "packaged": 0.40}

# V_i primary: Nutri-Score grade mapped to [0,1] when the lookup succeeds.
NUTRISCORE_TO_V = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2}

# F_i band centers used when the ResNet supervised reference overrides the VLM.
BAND_CENTER = {"fresh": 0.85, "edible_soon": 0.50, "spoiled": 0.15}

DEFECT_VOCAB = ["wrinkling", "bruising", "discoloration", "leaking", "visible cut", "mold"]
SPOILED_KEYWORDS = ["mold", "fuzz", "rot", "leak"]
DEGRADED_KEYWORDS = ["wrink", "bruise", "spot", "brown", "discolor", "cut", "shrivel"]


def freshness_band(score):
    if score < 0.3:
        return "spoiled"
    if score < 0.7:
        return "edible_soon"
    return "fresh"


def fuse_freshness(vlm_score, vlm_condition, resnet_condition):
    """Fusion rule for F_i (answers the 'state the source' red item).

    Produce: the ResNet-18 classifier is the supervised reference. If the
    VLM's numeric freshness_score falls inside the ResNet-predicted band,
    keep the VLM score (it is finer-grained). If the two disagree, the
    ResNet band wins and F_i is set to that band's center. If no ResNet
    prediction is available, fall back to the VLM score alone.
    Packaged/bakery items are assigned F_i = 1.0 by convention (longer
    shelf life; they never enter the produce-freshness constraints).
    """
    score = min(max(float(vlm_score), 0.0), 1.0)
    if resnet_condition is None:
        return score, "vlm_only"
    if freshness_band(score) == resnet_condition:
        return score, "agree_vlm_score"
    return BAND_CENTER[resnet_condition], "resnet_override"


# ---------------------------------------------------------------------------
# Model wrapper (zero-shot, prompt-only: no training or fine-tuning is done)
# ---------------------------------------------------------------------------
class FoodVLM:
    def __init__(self, model_id, token):
        print(f"Loading {model_id} in 4-bit quantized, zero-shot mode...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.processor = LlavaProcessor.from_pretrained(model_id, token=token)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            token=token,
        )

    @torch.inference_mode()
    def assess(self, image, prompt):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.model.device
        )
        # Greedy decoding for reproducibility (was do_sample=True, T=0.2).
        output = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)
        text = self.processor.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        del inputs, output
        return text


# The prompt now requests a machine-readable "defects" list (fixed vocabulary)
# in addition to the free-text evidence, so the Python safety override can
# actually key off it.
PROMPT = (
    "USER: <image>\n"
    "ACT AS FOOD SAFETY INSPECTOR. Your goal is to find reasons to REJECT this item.\n\n"
    "STEP 1: ANALYZE THE IMAGE FOR DEFECTS\n"
    "- Do you see any white/green fuzz or grey furry patches? (mold)\n"
    "- Is there any liquid pooling or wet stickiness? (leaking)\n"
    "- Are there deep black pits, skin ruptures, or collapsed soft areas? (rot)\n"
    "- Are there wrinkles, brown bruises, discoloration, cuts, or shriveling?\n\n"
    "STEP 2: CLASSIFY BASED ON HIERARCHY\n"
    "1. If mold, leaking, or rot is found -> ALWAYS 'spoiled'.\n"
    "2. If skin is dry but wrinkled, bruised, discolored, or shriveled -> 'edible_soon'.\n"
    "3. If skin is perfectly taut, shiny, and vibrant -> 'fresh'.\n\n"
    "Output ONLY raw JSON with these exact keys:\n"
    '{"food_name": "string", "food_type": "produce|packaged|bakery", '
    '"condition": "fresh|edible_soon|spoiled", '
    '"defects": ["wrinkling|bruising|discoloration|leaking|visible cut|mold", ...] '
    "(empty list if none), "
    '"evidence": "describe the specific defects found", '
    '"freshness_score": float between 0.0 and 1.0}\n'
    "ASSISTANT:"
)


# ---------------------------------------------------------------------------
# Nutrition lookup with explicit hit/fallback accounting
# ---------------------------------------------------------------------------
nutrition_cache = {}
lookup_stats = {"attempted": 0, "matched": 0, "fallback": 0}


def get_nutrition_data(food_name, contact_email, retries=2):
    food_name = (food_name or "").lower().strip()
    if not food_name or food_name in ("unknown", "food"):
        return None
    if food_name in nutrition_cache:
        return nutrition_cache[food_name]

    lookup_stats["attempted"] += 1
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 3,
    }
    headers = {"User-Agent": f"FoodResearchApp/1.0 (contact: {contact_email})"}

    for _ in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("products"):
                    product = next(
                        (p for p in data["products"] if "nutriments" in p),
                        data["products"][0],
                    )
                    n = product.get("nutriments", {})
                    result = {
                        "matched": True,
                        "source_name": product.get("product_name", food_name),
                        "calories_100g": n.get("energy-kcal_100g"),
                        "proteins_100g": n.get("proteins_100g"),
                        "fat_100g": n.get("fat_100g"),
                        "fiber_100g": n.get("fiber_100g"),
                        "nutriscore": str(product.get("nutriscore_grade", "")).upper() or None,
                    }
                    nutrition_cache[food_name] = result
                    lookup_stats["matched"] += 1
                    return result
            break
        except requests.RequestException:
            time.sleep(1)
    lookup_stats["fallback"] += 1
    return {"matched": False, "note": "No API data found"}


def nutrition_value(nutrition_info, category):
    """Scalar V_i in [0,1] (answers the 'state the exact formula' red item).

    Primary: Nutri-Score grade A..E mapped linearly to 1.0..0.2.
    Secondary (grade missing but macros present): a weighted sum of nutrient
    densities per 100 g, clipped to [0,1]:
        V_i = clip(0.5*protein/20 + 0.3*fiber/10 + 0.2*kcal/400, 0, 1)
    Fallback (no lookup match): category default
    (produce 0.85, bakery 0.60, packaged 0.40).
    """
    if nutrition_info and nutrition_info.get("matched"):
        grade = nutrition_info.get("nutriscore")
        if grade in NUTRISCORE_TO_V:
            return NUTRISCORE_TO_V[grade], "nutriscore"
        protein = nutrition_info.get("proteins_100g")
        fiber = nutrition_info.get("fiber_100g")
        kcal = nutrition_info.get("calories_100g")
        if any(v is not None for v in (protein, fiber, kcal)):
            v = (
                0.5 * (float(protein or 0) / 20.0)
                + 0.3 * (float(fiber or 0) / 10.0)
                + 0.2 * (float(kcal or 0) / 400.0)
            )
            return round(min(max(v, 0.0), 1.0), 4), "macros"
    return CATEGORY_DEFAULT_NUTRITION.get(category, 0.40), "category_default"


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group().replace(r"\_", "_"))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", default="food_original")
    ap.add_argument("--yolo-json", default="expert_predictions.json")
    ap.add_argument("--resnet-json", default=None,
                    help="optional {file_name: fresh|edible_soon|spoiled} predictions "
                         "from the ResNet-18 supervised reference")
    ap.add_argument("--output", default="llava_assessments.json")
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--contact-email", default=os.environ.get("OFF_CONTACT_EMAIL", ""))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable (never hardcode tokens).")

    torch.manual_seed(SEED)

    vlm = FoodVLM(args.model_id, token)

    with open(args.yolo_json) as f:
        yolo_raw = json.load(f)
    yolo_lookup = {k: v["category"].lower() for k, v in yolo_raw.items()}

    resnet_lookup = {}
    if args.resnet_json and os.path.exists(args.resnet_json):
        with open(args.resnet_json) as f:
            resnet_lookup = {k: v.lower() for k, v in json.load(f).items()}

    results = []
    print(f"Starting zero-shot analysis on {len(yolo_lookup)} images (seed={SEED})...")

    for i, file_name in enumerate(tqdm(sorted(yolo_lookup))):
        img_path = os.path.join(args.image_dir, file_name)
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            yolo_cat = yolo_lookup.get(file_name, "unknown")

            response_text = vlm.assess(image, PROMPT)
            image.close()
            parsed = extract_json(response_text)
            if not parsed:
                results.append({"file_name": file_name, "yolo_type": yolo_cat,
                                "parse_failed": True, "raw": response_text[:500]})
                continue

            # --- Safety override (now actually reachable: the prompt asks for
            # a "defects" list, and the evidence string is scanned as well) ---
            defect_text = " ".join(
                str(d).lower() for d in parsed.get("defects", []) or []
            ) + " " + str(parsed.get("evidence", "")).lower()
            if any(k in defect_text for k in SPOILED_KEYWORDS):
                parsed["condition"] = "spoiled"
            elif any(k in defect_text for k in DEGRADED_KEYWORDS):
                if parsed.get("condition") == "fresh":
                    parsed["condition"] = "edible_soon"

            category = yolo_cat if yolo_cat in CATEGORY_WEIGHTS_LB else \
                str(parsed.get("food_type", "packaged")).lower()

            # --- Nutrition: API lookup for packaged/bakery, category default
            # for generic produce (OpenFoodFacts is keyed to barcoded goods) ---
            if category in ("packaged", "bakery"):
                parsed["nutrition_info"] = get_nutrition_data(
                    parsed.get("food_name"), args.contact_email
                )
            else:
                parsed["nutrition_info"] = {"matched": False,
                                            "note": "produce — category default"}

            # --- Optimizer inputs: V_i, F_i, W_i (Section II-E) ---
            v_i, v_source = nutrition_value(parsed["nutrition_info"], category)
            if category == "produce":
                f_i, f_source = fuse_freshness(
                    parsed.get("freshness_score", 0.5),
                    parsed.get("condition"),
                    resnet_lookup.get(file_name),
                )
            else:
                f_i, f_source = 1.0, "nonperishable_convention"

            parsed.update(
                file_name=file_name,
                yolo_type=yolo_cat,
                category=category,
                V_i=v_i, V_source=v_source,
                F_i=round(f_i, 4), F_source=f_source,
                W_i=CATEGORY_WEIGHTS_LB.get(category, 1.0),
            )
            results.append(parsed)

            if len(results) % 10 == 0:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
            if i % 20 == 0:
                torch.cuda.empty_cache()
        except Exception as e:  # keep the batch alive on single-image failures
            print(f"\nError processing {file_name}: {e}")

    attempted = lookup_stats["attempted"]
    summary = {
        "seed": SEED,
        "decoding": "greedy (do_sample=False)",
        "zero_shot": True,
        "n_images": len(results),
        "openfoodfacts": {
            **lookup_stats,
            "match_rate": round(lookup_stats["matched"] / attempted, 4) if attempted else None,
        },
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)
    print(f"\nRun complete. {json.dumps(summary, indent=2)}\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
