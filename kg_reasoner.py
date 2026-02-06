# kg_reasoner.py
import json
import numpy as np


class KnowledgeGraph:
    def __init__(self, path="kg.json"):
        with open(path, "r") as f:
            self.kg = json.load(f)

    # -------------------------------------------------------
    # MIRRORING FOR LEFT-HANDED SHOOTERS
    # -------------------------------------------------------
    def mirror_biomechanics(self, bm, handedness):
        if handedness.lower().startswith("right"):
            return bm

        mirrored = dict(bm)

        swap_pairs = [
            ("right_shoulder_angle_deg", "left_shoulder_angle_deg"),
            ("right_elbow_angle_deg", "left_elbow_angle_deg"),
            ("right_knee_angle_deg", "left_knee_angle_deg"),
        ]

        for a, b in swap_pairs:
            if a in mirrored and b in mirrored:
                mirrored[a], mirrored[b] = mirrored[b], mirrored[a]

        return mirrored

    # -------------------------------------------------------
    # MATCH USER TO ELITE SHOOTER MANIFOLDS
    # -------------------------------------------------------
    def match_elite_shooters(self, user_features, top_k=3):
        distances = []

        for sid, shooter in self.kg["elite_shooters"].items():
            mu = []
            user = []

            for feat in self.kg["feature_nodes"]:
                if feat in shooter["manifold"] and feat in user_features:
                    mu.append(float(shooter["manifold"][feat]))
                    user.append(float(user_features[feat]))

            if len(mu) < 3:
                continue

            mu = np.array(mu)
            user = np.array(user)

            dist = np.linalg.norm(user - mu)
            distances.append((sid, shooter["display_name"], float(dist)))

        distances.sort(key=lambda x: x[2])
        return distances[:top_k]

    # -------------------------------------------------------
    # COLLECT ACTIVE STYLE CONSTRAINTS
    # -------------------------------------------------------
    def get_active_style_constraints(self, matched_shooters):
        active_constraints = {}

        for sid, _, _ in matched_shooters:
            traits = self.kg["elite_shooters"][sid].get("style_traits", [])

            for trait in traits:
                if trait in self.kg["style_constraints"]:
                    active_constraints[trait] = self.kg["style_constraints"][trait]

        return active_constraints

    # -------------------------------------------------------
    # DETECT LOCAL DEVIATIONS (STYLE-AWARE)
    # -------------------------------------------------------
    def evaluate_local_deviations(self, user_features, constraints):
        deviations = []

        for style, rule in constraints.items():
            requires = rule.get("requires", {})

            for feat, cond in requires.items():
                if feat not in user_features:
                    continue

                val = float(user_features[feat])

                if "min" in cond and val < cond["min"]:
                    deviations.append({
                        "feature": feat,
                        "issue": "below_style_min",
                        "style": style,
                        "value": round(val, 2),
                        "threshold": cond["min"]
                    })

                if "max" in cond and val > cond["max"]:
                    deviations.append({
                        "feature": feat,
                        "issue": "above_style_max",
                        "style": style,
                        "value": round(val, 2),
                        "threshold": cond["max"]
                    })

        return deviations

    # -------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------
    def evaluate_all(self, biomechanics_dict, handedness="right-handed"):
        bm_raw = self.mirror_biomechanics(biomechanics_dict, handedness)
        bm = self.normalize_user_features(bm_raw)

        matched = self.match_elite_shooters(bm)
        constraints = self.get_active_style_constraints(matched)
        deviations = self.evaluate_local_deviations(bm, constraints)

        return {
            "matched_shooters": matched,
            "active_styles": list(constraints.keys()),
            "local_deviations": deviations
        }
    
    def normalize_user_features(self, bm):
        """
        Map pipeline biomechanical outputs to KG feature names.
        """
        return {
            "elbow_angle_release_deg": bm["angles_at_release"]["elbow"],
            "knee_angle_release_deg": bm["angles_at_release"]["knee"],
            "shoulder_angle_release_deg": bm["angles_at_release"]["shoulder"],
            "dip_duration_s": bm["timing"]["dip_duration_s"],
            "drive_duration_s": bm["timing"]["drive_duration_s"],
            "knee_elbow_delay_ms": bm["coordination"]["knee_to_elbow_delay_s"] * 1000,
            "elbow_rom_deg": bm["range_of_motion"]["elbow_extension"],
            "knee_rom_deg": bm["range_of_motion"]["knee_extension"],
            "elbow_smoothness": bm["consistency"]["elbow_smoothness"],
            "knee_smoothness": bm["consistency"].get("knee_smoothness", 0.0)
        }

