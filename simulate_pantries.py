import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, ConfusionMatrixDisplay)

# ── PuLP import ────────────────────────────────────────────────────────────
try:
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                      value, PULP_CBC_CMD)
    USE_PULP = True
except ImportError:
    USE_PULP = False
    print("[INFO] PuLP not found — using greedy fallback (same 5 rules apply).")

# ==============================================================================
# 1. PARAMETERS
# ==============================================================================
W_FRESHNESS  = 3
W_PERISHABLE = 2
# Utility = 3*F^2 + 2*P + N

FRESHNESS_DISCARD_MAX = 0.30
FRESHNESS_LOCAL_MAX   = 0.70
LOCAL_DISTANCE_LIMIT  = 15.0
FRESH_DISTANCE_LIMIT  = 50.0

FOOD_TYPES        = ["leafy greens","berries","dairy/eggs","root veg","citrus",
                     "cooked meals","bread/bakery","canned/pkg","dry goods"]
PERISHABILITY_9   = [2,2,2,1,1,1,1,0,0]
NUTRITION_CONST_9 = [0.85]*7 + [0.40,0.40]

# Estimated kg per item by food type (for impact metrics)
KG_PER_ITEM = [0.3,0.2,0.5,0.4,0.3,0.5,0.4,0.4,0.5]
# Estimated servings per kg
SERVINGS_PER_KG = 4
# kg CO2 saved per kg food rescued (WRAP estimate)
CO2_PER_KG = 2.5
# Average $ value per kg food
DOLLAR_PER_KG = 3.5

# ==============================================================================
# 2. PANTRY DEFINITIONS
# ==============================================================================
pantries = {
    "Pantry_A": {"distance": 15, "capacity": 21},
    "Pantry_B": {"distance": 25, "capacity":  3},
    "Pantry_C": {"distance": 15, "capacity": 15},
}

ITEMS_PER_PANTRY = 30

# ==============================================================================
# 3. ITEM GENERATION + GROUND TRUTH SIMULATION
# ==============================================================================
random.seed(42)
np.random.seed(42)

items = {}
for src_pantry in pantries:
    for j in range(ITEMS_PER_PANTRY):
        iid       = f"{src_pantry}__item{j:02d}"
        class_idx = random.randint(0, 8)

        # Ground truth freshness (what the item actually is)
        true_freshness = float(np.clip(np.random.beta(3, 2), 0, 1))
        if random.random() < 0.10:
            true_freshness = round(random.uniform(0.0, 0.28), 3)
        true_freshness = round(true_freshness, 3)

        # Predicted freshness (VLM/ResNet output — add realistic noise)
        noise          = np.random.normal(0, 0.08)
        pred_freshness = float(np.clip(true_freshness + noise, 0, 1))
        pred_freshness = round(pred_freshness, 3)

        items[iid] = {
            "true_freshness":  true_freshness,
            "freshness":       pred_freshness,      # what the model sees
            "food_type":       FOOD_TYPES[class_idx],
            "class_idx":       class_idx,
            "perishability":   PERISHABILITY_9[class_idx],
            "nutrition":       NUTRITION_CONST_9[class_idx],
            "kg":              KG_PER_ITEM[class_idx],
            "source":          src_pantry,
        }

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def compute_utility(item):
    return (W_FRESHNESS  * item["freshness"]**2
          + W_PERISHABLE * item["perishability"]
          + item["nutrition"])

def freshness_band(f):
    if f > FRESHNESS_LOCAL_MAX:   return "fresh"
    if f > FRESHNESS_DISCARD_MAX: return "edible_soon"
    return "spoiled"

# ==============================================================================
# 5. LP / GREEDY OPTIMISATION  (your exact 5-rule structure)
# ==============================================================================
def run_optimiser(items_dict, pantries_dict):
    """Returns decisions dict: item_id -> pantry name or 'Compost'."""
    if USE_PULP:
        prob    = LpProblem("Food_Bank_Routing", LpMaximize)
        choices = {i: {p: LpVariable(f"x_{i}_{p}", cat="Binary")
                       for p in pantries_dict}
                   for i in items_dict}
        prob += lpSum(choices[i][p] * compute_utility(items_dict[i])
                      for i in items_dict for p in pantries_dict)
        for i in items_dict:
            f = items_dict[i]["freshness"]
            prob += lpSum([choices[i][p] for p in pantries_dict]) <= 1   # Rule 1
            if f <= FRESHNESS_DISCARD_MAX:                                # Rule 2
                for p in pantries_dict: prob += choices[i][p] == 0
            if FRESHNESS_DISCARD_MAX < f <= FRESHNESS_LOCAL_MAX:          # Rule 3
                for p in pantries_dict:
                    if pantries_dict[p]["distance"] > LOCAL_DISTANCE_LIMIT:
                        prob += choices[i][p] == 0
            if f > FRESHNESS_LOCAL_MAX:                                   # Rule 4
                for p in pantries_dict:
                    if pantries_dict[p]["distance"] > FRESH_DISTANCE_LIMIT:
                        prob += choices[i][p] == 0
        for p in pantries_dict:                                           # Rule 5
            prob += lpSum([choices[i][p] for i in items_dict]) <= pantries_dict[p]["capacity"]
        prob.solve(PULP_CBC_CMD(msg=0))
        dec = {}
        for i in items_dict:
            assigned = next((p for p in pantries_dict if value(choices[i][p]) == 1), None)
            dec[i] = assigned if assigned else "Compost"
        return dec
    else:
        slot_used = {p: 0 for p in pantries_dict}
        dec = {}
        def elig(iid):
            f = items_dict[iid]["freshness"]
            if f <= FRESHNESS_DISCARD_MAX: return []
            out = []
            for p, pd in pantries_dict.items():
                if FRESHNESS_DISCARD_MAX < f <= FRESHNESS_LOCAL_MAX:
                    if pd["distance"] > LOCAL_DISTANCE_LIMIT: continue
                if f > FRESHNESS_LOCAL_MAX:
                    if pd["distance"] > FRESH_DISTANCE_LIMIT: continue
                out.append(p)
            return out
        for iid in sorted(items_dict, key=lambda i: -compute_utility(items_dict[i])):
            cands = [p for p in elig(iid) if slot_used[p] < pantries_dict[p]["capacity"]]
            if cands:
                chosen = max(cands, key=lambda p: pantries_dict[p]["capacity"] - slot_used[p])
                dec[iid] = chosen
                slot_used[chosen] += 1
            else:
                dec[iid] = "Compost"
        return dec

# ── Run optimiser ──────────────────────────────────────────────────────────
decisions = run_optimiser(items, pantries)

# ── Baseline: naive random assignment (ignores utility, same rules) ────────
def run_baseline(items_dict, pantries_dict):
    random.seed(99)
    slot_used = {p: 0 for p in pantries_dict}
    dec = {}
    shuffled = list(items_dict.keys())
    random.shuffle(shuffled)
    for iid in shuffled:
        f = items_dict[iid]["freshness"]
        if f <= FRESHNESS_DISCARD_MAX:
            dec[iid] = "Compost"; continue
        cands = []
        for p, pd in pantries_dict.items():
            if FRESHNESS_DISCARD_MAX < f <= FRESHNESS_LOCAL_MAX:
                if pd["distance"] > LOCAL_DISTANCE_LIMIT: continue
            if f > FRESHNESS_LOCAL_MAX:
                if pd["distance"] > FRESH_DISTANCE_LIMIT: continue
            if slot_used[p] < pd["capacity"]:
                cands.append(p)
        if cands:
            chosen = random.choice(cands)
            dec[iid] = chosen
            slot_used[chosen] += 1
        else:
            dec[iid] = "Compost"
    return dec

baseline_decisions = run_baseline(items, pantries)

# ==============================================================================
# 6. BUILD RESULTS DATAFRAME
# ==============================================================================
rows = []
for iid, dest in decisions.items():
    item = items[iid]
    rows.append({
        "id":              iid,
        "source":          item["source"],
        "food_type":       item["food_type"],
        "class_idx":       item["class_idx"],
        "true_freshness":  item["true_freshness"],
        "freshness":       item["freshness"],
        "perishability":   item["perishability"],
        "nutrition":       item["nutrition"],
        "kg":              item["kg"],
        "true_band":       freshness_band(item["true_freshness"]),
        "pred_band":       freshness_band(item["freshness"]),
        "band":            freshness_band(item["freshness"]),
        "utility":         round(compute_utility(item), 4),
        "decision":        dest,
        "donated":         dest != "Compost",
        "baseline_dec":    baseline_decisions[iid],
        "baseline_donated":baseline_decisions[iid] != "Compost",
    })

df      = pd.DataFrame(rows)
df_don  = df[df["donated"]]
df_comp = df[~df["donated"]]
p_names = list(pantries.keys())
donated_count = {p: len(df[df["decision"] == p]) for p in pantries}

# ==============================================================================
# 7. ACCURACY METRICS
# ==============================================================================
# Freshness band classification accuracy (predicted band vs true band)
band_labels   = ["fresh","edible_soon","spoiled"]
y_true_band   = df["true_band"].tolist()
y_pred_band   = df["pred_band"].tolist()
cm_band       = confusion_matrix(y_true_band, y_pred_band, labels=band_labels)
band_report   = classification_report(y_true_band, y_pred_band,
                                      labels=band_labels, output_dict=True)
band_accuracy = np.mean([t == p for t, p in zip(y_true_band, y_pred_band)])

# Donate/Compost routing accuracy
# Ground truth: what decision WOULD be made on true freshness
true_decisions = run_optimiser(
    {k: {**v, "freshness": v["true_freshness"]} for k, v in items.items()},
    pantries
)
y_true_route = [1 if true_decisions[i] != "Compost" else 0 for i in df["id"]]
y_pred_route = [1 if decisions[i]      != "Compost" else 0 for i in df["id"]]
cm_route     = confusion_matrix(y_true_route, y_pred_route)
route_report = classification_report(y_true_route, y_pred_route,
                                     target_names=["Compost","Donate"],
                                     output_dict=True)
route_accuracy = np.mean([t == p for t, p in zip(y_true_route, y_pred_route)])

# ROC curve
roc_fpr, roc_tpr, _ = roc_curve(y_true_route, df["freshness"])
roc_auc = auc(roc_fpr, roc_tpr)

# Baseline utility vs optimiser utility
opt_utility  = sum(compute_utility(items[i]) for i in decisions  if decisions[i]  != "Compost")
base_utility = sum(compute_utility(items[i]) for i in baseline_decisions if baseline_decisions[i] != "Compost")
utility_gain = 100 * (opt_utility - base_utility) / max(base_utility, 1)

opt_donated  = sum(1 for v in decisions.values()          if v != "Compost")
base_donated = sum(1 for v in baseline_decisions.values() if v != "Compost")

# Impact metrics (optimiser)
donated_kg    = df_don["kg"].sum()
donated_meals = donated_kg * SERVINGS_PER_KG
co2_saved     = donated_kg * CO2_PER_KG
dollar_saved  = donated_kg * DOLLAR_PER_KG
wasted_kg     = df_comp["kg"].sum()

# ==============================================================================
# 8. CONSOLE SUMMARY
# ==============================================================================
print("=" * 60)
print("  Food Bank Distribution  |  Utility = 3·F² + 2·P + N")
print("=" * 60)
print(f"  Total items          : {len(df)}")
print(f"  Donated              : {len(df_don)}  ({100*len(df_don)/len(df):.1f}%)")
print(f"  Composted            : {len(df_comp)}  ({100*len(df_comp)/len(df):.1f}%)")
print()
print(f"  Band classification accuracy : {band_accuracy:.1%}")
print(f"  Routing accuracy             : {route_accuracy:.1%}")
print(f"  ROC AUC                      : {roc_auc:.3f}")
print(f"  Utility gain vs baseline     : +{utility_gain:.1f}%")
print()
print(f"  Estimated kg rescued   : {donated_kg:.1f} kg")
print(f"  Estimated meals        : {donated_meals:.0f} servings")
print(f"  CO₂ saved              : {co2_saved:.1f} kg CO₂")
print(f"  Dollar value rescued   : ${dollar_saved:.2f}")
print("=" * 60)

# ==============================================================================
# 9. FIGURE 1 — ORIGINAL DISTRIBUTION PLOTS  (6 panels)
# ==============================================================================
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE  = {"Donate": "#4CAF50", "Compost": "#F44336"}
P_COLORS = {"Pantry_A": "#5B9BD5", "Pantry_B": "#70AD47", "Pantry_C": "#ED7D31"}
BAND_PAL = {"fresh": "#4CAF50", "edible_soon": "#FFC107", "spoiled": "#F44336"}

df["dec_label"] = df["decision"].apply(lambda x: "Donate" if x != "Compost" else "Compost")

fig1 = plt.figure(figsize=(20, 14))
gs1  = GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

# A: Freshness histogram
ax_a = fig1.add_subplot(gs1[0, 0])
for lab, grp in df.groupby("dec_label"):
    ax_a.hist(grp["freshness"], bins=20, alpha=0.65, color=PALETTE[lab], label=lab)
ax_a.axvline(FRESHNESS_DISCARD_MAX, color="red",    ls="--", lw=1.4, label="Rule 2: discard ≤0.30")
ax_a.axvline(FRESHNESS_LOCAL_MAX,   color="orange", ls="--", lw=1.4, label="Rule 3: local ≤0.70")
ax_a.set_title("Freshness Distribution by Decision")
ax_a.set_xlabel("Freshness Score"); ax_a.set_ylabel("Count")
ax_a.legend(fontsize=8)

# B: Utility by freshness band
ax_b = fig1.add_subplot(gs1[0, 1])
sns.boxplot(data=df, x="band", y="utility", order=["fresh","edible_soon","spoiled"],
            hue="band", palette=BAND_PAL, ax=ax_b, width=0.5, legend=False)
ax_b.set_title("Utility Score by Freshness Band\n(3·F² + 2·P + N)")
ax_b.set_xlabel("Freshness Band"); ax_b.set_ylabel("Utility Score")

# C: Freshness per pantry (donated only)
ax_c = fig1.add_subplot(gs1[0, 2])
if len(df_don):
    sns.boxplot(data=df_don, x="decision", y="freshness", order=p_names,
                hue="decision", palette=P_COLORS, ax=ax_c, width=0.5, legend=False)
    for idx, p in enumerate(p_names):
        d = pantries[p]["distance"]
        rule = "edible+fresh" if d <= LOCAL_DISTANCE_LIMIT else "fresh only"
        ax_c.text(idx, 1.02, f"{d} mi ({rule})", ha="center", fontsize=7,
                  color="gray", style="italic")
    ax_c.set_title("Freshness of Donated Items per Pantry")
    ax_c.set_xlabel(""); ax_c.set_ylabel("Freshness Score")
    ax_c.set_ylim(0, 1.10); ax_c.tick_params(axis="x", rotation=10)

# D: Where food was distributed
ax_d = fig1.add_subplot(gs1[1, 0])
dest_labels = p_names + ["Compost"]
bands_order = ["fresh","edible_soon","spoiled"]
band_colors = ["#4CAF50","#FFC107","#F44336"]
bottom = np.zeros(len(dest_labels))
for band, bc in zip(bands_order, band_colors):
    heights = [len(df[(df["decision"] == d) & (df["band"] == band)]) for d in dest_labels]
    ax_d.bar(dest_labels, heights, bottom=bottom, color=bc, alpha=0.85,
             label=band.replace("_"," "), edgecolor="white", linewidth=0.6)
    bottom += np.array(heights, dtype=float)
for idx, total in enumerate(bottom):
    ax_d.text(idx, total + 0.3, str(int(total)), ha="center", fontsize=9, fontweight="bold")
ax_d.set_title("Where Food Was Distributed\n(stacked by freshness band)")
ax_d.set_ylabel("Item Count"); ax_d.legend(title="Band", fontsize=8)
ax_d.tick_params(axis="x", rotation=10)

# E: Capacity utilisation
ax_e = fig1.add_subplot(gs1[1, 1])
x     = np.arange(len(p_names))
caps  = [pantries[p]["capacity"] for p in p_names]
used  = [donated_count[p]        for p in p_names]
fracs = [u/c for u, c in zip(used, caps)]
bars_e = ax_e.bar(x, fracs, color=[P_COLORS[p] for p in p_names],
                  width=0.5, edgecolor="white", linewidth=0.8)
ax_e.axhline(1.0, color="red", ls="--", lw=1.2, label="Capacity full (Rule 5)")
for bar, frac, u, c in zip(bars_e, fracs, used, caps):
    ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f"{u}/{c}  ({frac:.0%})", ha="center", va="bottom", fontsize=9)
ax_e.set_xticks(x)
ax_e.set_xticklabels([f"{p}\n({pantries[p]['distance']} mi)" for p in p_names], rotation=0)
ax_e.set_ylim(0, 1.4); ax_e.set_title("Pantry Capacity Utilisation (Rule 5)")
ax_e.set_ylabel("Fraction of Available Slots Filled"); ax_e.legend(fontsize=8)

# F: Strip plot — freshness per destination
ax_f = fig1.add_subplot(gs1[1, 2])
dest_order      = p_names + ["Compost"]
dest_colors_map = {**P_COLORS, "Compost": "#F44336"}
for dest in dest_order:
    grp    = df[df["decision"] == dest]["freshness"]
    jitter = np.random.uniform(-0.15, 0.15, size=len(grp))
    y_pos  = dest_order.index(dest)
    ax_f.scatter(grp, [y_pos + j for j in jitter],
                 c=dest_colors_map[dest], alpha=0.55, s=25, edgecolors="none")
ax_f.axvline(FRESHNESS_DISCARD_MAX, color="red",    ls="--", lw=1.3,
             label=f"Rule 2: spoiled ≤{FRESHNESS_DISCARD_MAX}")
ax_f.axvline(FRESHNESS_LOCAL_MAX,   color="orange", ls="--", lw=1.3,
             label=f"Rule 3: edible_soon ≤{FRESHNESS_LOCAL_MAX}")
ax_f.set_yticks(range(len(dest_order))); ax_f.set_yticklabels(dest_order)
ax_f.set_xlabel("Freshness Score")
ax_f.set_title("Freshness Score per Destination\n(Rules 1–4 visible)")
ax_f.legend(fontsize=8, loc="lower right")

fig1.suptitle(
    "Food Bank Distribution Simulation\n"
    "Utility = 3·F² + 2·P + N  |  "
    "Rules: (1) one pantry max  (2) spoiled→discard  "
    "(3) edible_soon≤15 mi  (4) fresh≤50 mi  (5) capacity cap",
    fontsize=11, fontweight="bold", y=1.01
)
fig1.savefig("distribution_performance.png", dpi=150, bbox_inches="tight")
print("Saved → distribution_performance.png")

# ==============================================================================
# 10. FIGURE 2 — ACCURACY & IMPACT  (6 panels)
# ==============================================================================
fig2 = plt.figure(figsize=(20, 14))
gs2  = GridSpec(2, 3, figure=fig2, hspace=0.50, wspace=0.38)

# ── G: Confusion matrix — freshness band classification ──────────────────────
ax_g = fig2.add_subplot(gs2[0, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_band,
                              display_labels=["fresh","edible_soon","spoiled"])
disp.plot(ax=ax_g, colorbar=False, cmap="Greens")
ax_g.set_title(f"Freshness Band Classification\nConfusion Matrix  (Acc = {band_accuracy:.1%})")
ax_g.set_xlabel("Predicted Band"); ax_g.set_ylabel("True Band")

# ── H: Per-class F1 bar chart ────────────────────────────────────────────────
ax_h = fig2.add_subplot(gs2[0, 1])
classes  = ["fresh","edible_soon","spoiled"]
f1s      = [band_report[c]["f1-score"]  for c in classes]
precs    = [band_report[c]["precision"] for c in classes]
recalls  = [band_report[c]["recall"]    for c in classes]
xb = np.arange(len(classes)); w = 0.25
ax_h.bar(xb - w, precs,   width=w, label="Precision", color="#5B9BD5", alpha=0.85)
ax_h.bar(xb,     recalls, width=w, label="Recall",    color="#70AD47", alpha=0.85)
ax_h.bar(xb + w, f1s,     width=w, label="F1-Score",  color="#ED7D31", alpha=0.85)
for i, (p, r, f) in enumerate(zip(precs, recalls, f1s)):
    ax_h.text(i-w, p+0.01, f"{p:.2f}", ha="center", fontsize=8)
    ax_h.text(i,   r+0.01, f"{r:.2f}", ha="center", fontsize=8)
    ax_h.text(i+w, f+0.01, f"{f:.2f}", ha="center", fontsize=8)
ax_h.set_xticks(xb); ax_h.set_xticklabels(classes)
ax_h.set_ylim(0, 1.25); ax_h.set_ylabel("Score")
ax_h.set_title("Precision / Recall / F1\nper Freshness Band")
ax_h.legend(fontsize=9)

# ── I: ROC curve — donate vs compost ─────────────────────────────────────────
ax_i = fig2.add_subplot(gs2[0, 2])
ax_i.plot(roc_fpr, roc_tpr, color="#5B9BD5", lw=2,
          label=f"ROC curve (AUC = {roc_auc:.3f})")
ax_i.plot([0,1],[0,1], color="gray", ls="--", lw=1.2, label="Random classifier")
ax_i.fill_between(roc_fpr, roc_tpr, alpha=0.10, color="#5B9BD5")
ax_i.set_xlabel("False Positive Rate"); ax_i.set_ylabel("True Positive Rate")
ax_i.set_title("ROC Curve\nDonate vs Compost Decision")
ax_i.legend(fontsize=9); ax_i.set_xlim(0,1); ax_i.set_ylim(0,1.02)

# ── J: Confusion matrix — routing decision ────────────────────────────────────
ax_j = fig2.add_subplot(gs2[1, 0])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_route,
                               display_labels=["Compost","Donate"])
disp2.plot(ax=ax_j, colorbar=False, cmap="Blues")
ax_j.set_title(f"Donate / Compost Routing\nConfusion Matrix  (Acc = {route_accuracy:.1%})")
ax_j.set_xlabel("Predicted Decision"); ax_j.set_ylabel("True Decision")

# ── K: Optimiser vs baseline comparison ──────────────────────────────────────
ax_k = fig2.add_subplot(gs2[1, 1])
metrics  = ["Items Donated", "Total Utility", "kg Rescued"]
opt_vals  = [opt_donated,  round(opt_utility, 1),  round(donated_kg, 1)]
base_vals = [base_donated, round(base_utility, 1), round(df[df["baseline_donated"]]["kg"].sum(), 1)]
xk = np.arange(len(metrics)); wk = 0.32
b1 = ax_k.bar(xk - wk/2, base_vals, width=wk, label="Baseline (random)",
               color="#B0BEC5", alpha=0.85, edgecolor="white")
b2 = ax_k.bar(xk + wk/2, opt_vals,  width=wk, label="Optimiser (utility sort)",
               color="#4CAF50", alpha=0.85, edgecolor="white")
for bar, val in zip(b1, base_vals):
    ax_k.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
              str(val), ha="center", fontsize=9)
for bar, val in zip(b2, opt_vals):
    ax_k.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
              str(val), ha="center", fontsize=9, fontweight="bold")
ax_k.set_xticks(xk); ax_k.set_xticklabels(metrics)
ax_k.set_title(f"Optimiser vs Baseline\n(Utility gain: +{utility_gain:.1f}%)")
ax_k.set_ylabel("Value"); ax_k.legend(fontsize=9)

# ── L: Impact metrics bar ─────────────────────────────────────────────────────
ax_l = fig2.add_subplot(gs2[1, 2])
impact_labels = ["kg Rescued", "Meals\nProvided", "kg CO₂\nSaved", "Value\nSaved ($)"]
impact_vals   = [round(donated_kg,1), round(donated_meals,0),
                 round(co2_saved,1),  round(dollar_saved,1)]
impact_colors = ["#4CAF50","#5B9BD5","#26A69A","#FFA726"]
bars_l = ax_l.bar(impact_labels, impact_vals, color=impact_colors,
                  alpha=0.85, edgecolor="white", linewidth=0.8, width=0.55)
for bar, val in zip(bars_l, impact_vals):
    ax_l.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
              str(val), ha="center", fontsize=10, fontweight="bold")
ax_l.set_title("Estimated Real-World Impact\n(donated items)")
ax_l.set_ylabel("Amount"); ax_l.set_ylim(0, max(impact_vals)*1.2)

fig2.suptitle(
    "Model Accuracy & Real-World Impact\n"
    f"Band Accuracy: {band_accuracy:.1%}  |  "
    f"Routing Accuracy: {route_accuracy:.1%}  |  "
    f"AUC: {roc_auc:.3f}  |  "
    f"Utility gain vs baseline: +{utility_gain:.1f}%",
    fontsize=12, fontweight="bold", y=1.01
)
fig2.savefig("accuracy_and_impact.png", dpi=150, bbox_inches="tight")
print("Saved → accuracy_and_impact.png")

plt.show()
