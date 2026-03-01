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
            ("right_elbow_angle_deg",    "left_elbow_angle_deg"),
            ("right_knee_angle_deg",     "left_knee_angle_deg"),
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
            mu   = []
            user = []

            for feat in self.kg["feature_nodes"]:
                if feat in shooter["manifold"] and feat in user_features:
                    mu.append(float(shooter["manifold"][feat]))
                    user.append(float(user_features[feat]))

            if len(mu) < 3:
                continue

            mu   = np.array(mu)
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
        """
        For every active style constraint, check each required feature against
        its min/max bounds. When a violation is found the raw issue code is
        resolved to a human-readable deviation_label (from the style's
        deviation_labels map) and enriched with coaching_cue from the
        deviation_catalog.
        """
        deviations   = []
        catalog      = self.kg.get("deviation_catalog", {})

        for style, rule in constraints.items():
            requires        = rule.get("requires", {})
            deviation_labels = rule.get("deviation_labels", {})

            for feat, cond in requires.items():
                if feat not in user_features:
                    continue

                val = float(user_features[feat])

                if "min" in cond and val < cond["min"]:
                    label_key    = f"{feat}_below_min"
                    dev_label    = deviation_labels.get(label_key, "below_style_min")
                    catalog_entry = catalog.get(dev_label, {})

                    deviations.append({
                        "feature":      feat,
                        "issue":        "below_style_min",
                        "deviation":    dev_label,
                        "style":        style,
                        "value":        round(val, 2),
                        "threshold":    cond["min"],
                        "severity":     catalog_entry.get("severity", "unknown"),
                        "coaching_cue": catalog_entry.get("coaching_cue", "")
                    })

                if "max" in cond and val > cond["max"]:
                    label_key    = f"{feat}_above_max"
                    dev_label    = deviation_labels.get(label_key, "above_style_max")
                    catalog_entry = catalog.get(dev_label, {})

                    deviations.append({
                        "feature":      feat,
                        "issue":        "above_style_max",
                        "deviation":    dev_label,
                        "style":        style,
                        "value":        round(val, 2),
                        "threshold":    cond["max"],
                        "severity":     catalog_entry.get("severity", "unknown"),
                        "coaching_cue": catalog_entry.get("coaching_cue", "")
                    })

        return deviations

    # -------------------------------------------------------
    # COLLECT ACTIVE PRESERVATION CONSTRAINTS
    # -------------------------------------------------------
    def get_preserve_constraints(self, active_styles):
        """
        Returns the subset of preserve_constraints that are relevant to the
        shooter's active styles. The coaching layer uses this to avoid
        generating corrections that would degrade already-good mechanics.

        Returns a dict keyed by preserve constraint name. Each entry includes:
          - description  : what to protect and why
          - quantified   : whether the pipeline can measure it
          - related_feature : the KG feature node it maps to (or null)
          - preserve_condition : the condition that must stay true
        """
        preserve_block  = self.kg.get("preserve_constraints", {})
        active_preserve = {}

        for name, entry in preserve_block.items():
            if name.startswith("_"):
                continue
            relevant_styles = entry.get("relevant_styles", [])
            if any(s in active_styles for s in relevant_styles):
                active_preserve[name] = {
                    "quantified":         entry.get("quantified"),
                    "related_feature":    entry.get("related_feature"),
                    "preserve_condition": entry.get("preserve_condition"),
                    "description":        entry.get("description", "")
                }

        return active_preserve

    # -------------------------------------------------------
    # RESOLVE PRESERVE VIOLATIONS
    # (flag if a preserved feature is also flagged as a deviation)
    # -------------------------------------------------------
    def check_preserve_conflicts(self, deviations, preserve_constraints):
        """
        Cross-references active deviations against active preservation
        constraints. If a coaching correction for a deviation would touch
        a preserved feature, it is flagged here so the coaching layer can
        temper its advice.

        Returns a list of conflict records.
        """
        conflicts = []

        preserved_features = {
            entry["related_feature"]
            for entry in preserve_constraints.values()
            if entry.get("related_feature")
        }

        for dev in deviations:
            if dev["feature"] in preserved_features:
                # Find which preserve constraint owns this feature
                owners = [
                    name for name, entry in preserve_constraints.items()
                    if entry.get("related_feature") == dev["feature"]
                ]
                conflicts.append({
                    "deviation":            dev["deviation"],
                    "feature":              dev["feature"],
                    "preserve_constraints": owners,
                    "note": (
                        f"Correction for '{dev['deviation']}' on '{dev['feature']}' "
                        f"conflicts with preservation of {owners}. "
                        "Coaching cue should be framed carefully to improve the "
                        "deviation without disrupting the preserved mechanic."
                    )
                })

        return conflicts

    # -------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------
    def evaluate_all(self, biomechanics_dict, handedness="right-handed"):
        bm_raw = self.mirror_biomechanics(biomechanics_dict, handedness)
        bm     = self.normalize_user_features(bm_raw)

        matched     = self.match_elite_shooters(bm)
        constraints = self.get_active_style_constraints(matched)
        deviations  = self.evaluate_local_deviations(bm, constraints)

        active_styles    = list(constraints.keys())
        preserve         = self.get_preserve_constraints(active_styles)
        preserve_conflicts = self.check_preserve_conflicts(deviations, preserve)

        return {
            "matched_shooters":    matched,
            "active_styles":       active_styles,
            "local_deviations":    deviations,
            "preserve_constraints": preserve,
            "preserve_conflicts":  preserve_conflicts
        }

    # -------------------------------------------------------
    # FEATURE NORMALIZATION
    # -------------------------------------------------------
    def normalize_user_features(self, bm):
        """
        Map pipeline biomechanical outputs to KG feature names.
        """
        return {
            "elbow_angle_release_deg":   bm["angles_at_release"]["elbow"],
            "knee_angle_release_deg":    bm["angles_at_release"]["knee"],
            "shoulder_angle_release_deg": bm["angles_at_release"]["shoulder"],
            "dip_duration_s":            bm["timing"]["dip_duration_s"],
            "drive_duration_s":          bm["timing"]["drive_duration_s"],
            "knee_elbow_delay_ms":       bm["coordination"]["knee_to_elbow_delay_s"] * 1000,
            "elbow_rom_deg":             bm["range_of_motion"]["elbow_extension"],
            "knee_rom_deg":              bm["range_of_motion"]["knee_extension"],
            "elbow_smoothness":          bm["consistency"]["elbow_smoothness"],
            "knee_smoothness":           bm["consistency"].get("knee_smoothness", 0.0)
        }