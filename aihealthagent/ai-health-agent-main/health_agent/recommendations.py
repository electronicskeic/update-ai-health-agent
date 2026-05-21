from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    diet: list[str]
    fitness: list[str]
    notes: list[str] | None = None


def build_plan(*, bmi: float, category: str, risk_level: str, visceral_risk: str = "Normal", body_fat_pct: float | None = None) -> Plan:
    b = float(bmi)
    risk = str(risk_level)
    cat = str(category)

    diet: list[str] = []
    fitness: list[str] = []
    notes: list[str] = []

    # 1. Baseline ICMR-NIN (2024) Dietary Guidelines (Localized)
    diet.append("Prioritize complex, low-glycemic index (GI) carbohydrates like Jowar, Bajra, Ragi, or whole-wheat over white rice and maida.")
    diet.append("Maintain an adequate daily dietary fiber intake of at least 30g by consuming green leafy vegetables, seasonal fruits, and whole pulses.")
    diet.append("Restrict added cooking oils/visible fats to under 25-30g per day, choosing healthy fats (e.g. cold-pressed mustard, sesame, or groundnut oil).")
    diet.append("Limit sugar and salt intake (<5g of salt/day, as endorsed by ICMR-NIN) to protect cardiovascular and metabolic health.")

    # 2. Specific Obesity / BMI Heuristics
    if b >= 23.0: # South Asian threshold for Overweight is 23.0 (ICMR)
        diet.append("Apply the ICMR Plate Method: fill half of your plate with vegetables/salads, a quarter with lean protein, and a quarter with complex carbs.")
        diet.append("Swap deep-fried snacks for high-fiber, high-protein alternatives like roasted chana, sprouts, or direct boiled legumes.")
    elif b < 18.5:
        diet.append("Introduce nutrient-dense, calorie-rich Indian snacks such as almonds, walnuts, paneer, or roasted peanuts.")
        diet.append("Increase protein intake using healthy local sources (paneer, dals, curd, or lean meats) alongside complex carbs at every meal.")

    # 3. Visceral Fat & Body Composition Heuristics
    if visceral_risk in ("Increased Risk", "High Visceral Risk"):
        diet.append(f"⚠️ **Visceral Fat Focus**: Due to elevated abdominal fat indicators ({visceral_risk}), strictly avoid ultra-processed foods, trans fats, and sweetened beverages to prevent fatty liver risk.")
    
    if body_fat_pct is not None:
        if body_fat_pct > 25.0:
            diet.append(f"Target body re-composition: focus on high-quality proteins (1.2g/kg body weight) to preserve lean mass while oxidizing body fat.")

    # 4. Fitness / Activity Guidelines (ICMR Aligned)
    fitness.append("Aim for at least 150 minutes of moderate-intensity aerobic activity (brisk walking, cycling) per week, split across 5 days (30 mins/day).")
    fitness.append("Perform strength training sessions 2-3 times per week (using bodyweight, resistance bands, or light weights) to build muscle mass.")
    fitness.append("Build muscle mass specifically, as skeletal muscle is the main site for insulin-mediated glucose disposal and combatting South Asian metabolic syndrome.")

    if risk.lower() == "high":
        notes.append("Advisory: Since your calculated dataset risk is High, start with low-intensity activities (like walking) and increase duration slowly.")
    
    notes.append(f"Guidelines aligned with ICMR & National Institute of Nutrition (NIN) standards. Category: {cat}.")
    return Plan(diet=diet, fitness=fitness, notes=notes or None)


