import time
import random
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from ai_core.lm import run_local_lm
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class GoalManager:
    def __init__(self, decider: 'Decider'):
        self.decider = decider
        self.last_goal_management_time = 0

    def create_goal(self, content: str):
        """Autonomously create a new GOAL memory."""
        # DEDUPLICATION: Don't create the same goal twice
        existing = self.decider.memory_store.get_active_by_type("GOAL")
        if any(content.lower() in g[2].lower() for g in existing):
            self.decider.log(f"🎯 Decider: Goal already exists, skipping: {content}")
            return None

        # Deterministic identity for goals
        identity = self.decider.memory_store.compute_identity(content, "GOAL")

        mid = self.decider.memory_store.add_entry(
            identity=identity,
            text=content,
            mem_type="GOAL",
            subject="Assistant",
            confidence=1.0,
            source="decider_autonomous"
        )
        self.decider.log(f"🎯 Decider autonomously created GOAL (ID: {mid}): {content}")
        if hasattr(self.decider.meta_memory_store, 'add_event'):
            self.decider.meta_memory_store.add_event("GOAL_CREATED", "Assistant", f"Created Goal: {content}")
        return mid

    def run_autonomous_cycle(self):
        """
        Called by AICore when system is idle but has goals.
        Checks for active goals and executes tools to advance them.
        """
        # 1. Check for Active Goals
        stats = self.decider.memory_store.get_memory_stats()
        if stats.get('active_goals', 0) == 0:
            return None # Nothing to do

        # 2. Pick a goal, prioritizing LEAF goals (actionable) over ROOT goals (planning)
        with self.decider.memory_store._connect() as con:
            # Fetch more candidates to ensure we find leaf goals
            goals = con.execute("SELECT id, text, parent_id FROM memories WHERE type='GOAL' AND completed=0 ORDER BY id DESC LIMIT 20").fetchall()

        if not goals: return None

        # Separate into Leaf (has parent) and Root (no parent)
        leaf_goals = [g for g in goals if g[2] is not None]
        root_goals = [g for g in goals if g[2] is None]

        target_goal = None

        # 80% chance to pick a leaf goal if available (Action bias)
        if leaf_goals and random.random() < 0.8:
            target_goal = random.choice(leaf_goals)
        elif root_goals:
            target_goal = random.choice(root_goals)
        elif leaf_goals:
            target_goal = random.choice(leaf_goals)

        if not target_goal: return None

        goal = target_goal
        goal_id, goal_text, parent_id = goal

        # 2.5. The Architect: Check if goal needs decomposition
        # Only decompose ROOT goals (parent_id is None) to prevent infinite recursion
        if parent_id is None:
            # We check if it already has children
            with self.decider.memory_store._connect() as con:
                has_children = con.execute("SELECT 1 FROM memories WHERE parent_id = ? LIMIT 1", (goal_id,)).fetchone()

            if not has_children:
                if self._decompose_goal(goal_text, goal_id):
                    return None # Decomposition happened, wait for next cycle to pick up sub-goals

        # 3. SELF-PROMPT: "We have an active goal..."
        # We simulate a 'System' message to trigger the LLM to use a tool
        system_injection = f"[SYSTEM_TRIGGER]: You are idle. Active Goal: '{goal_text}'. Execute a tool (SEARCH/WIKI) to advance this."

        self.decider.log(f"🤖 Decider: Autonomously pursuing goal: {goal_text}")

        # 4. Run the thought loop (This triggers [EXECUTE: ...])
        response = run_local_lm(
            messages=[{"role": "system", "content": system_injection}],
            system_prompt=self.decider.get_settings().get("system_prompt")
        )

        return response

    def _decompose_goal(self, goal_text, goal_id):
        """Breaks a big goal into small steps."""
        # Only decompose if it looks complex (heuristic: length > 20 chars)
        if len(goal_text) < 20: return False

        # HTN (Hierarchical Task Network) Decomposition
        prompt = (
            f"Goal: '{goal_text}'\n"
            "Decompose this goal into a Hierarchical Task Network (HTN) tree structure.\n"
            "Return a JSON array of objects, where each object represents a high-level sub-task:\n"
            "[\n"
            "  {\"step\": \"Step description\", \"success_criteria\": \"Specific verifiable condition\"},\n"
            "  ...\n"
            "]\n"
            "Limit to 3-5 high-level steps."
        )
        settings = self.decider.get_settings()

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a strategic planner.",
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        try:
            steps = parse_json_array_loose(response)
        except Exception as e:
            self.decider.log(f"⚠️ Goal decomposition JSON parsing failed: {e}. Response: {response[:100]}...")
            return False

        if steps:
            self.decider.log(f"🏗️ The Architect: Breaking goal '{goal_text}' into {len(steps)} steps.")
            for step in steps:
                if isinstance(step, dict) and "step" in step:
                    text = f"{step['step']} [Criteria: {step.get('success_criteria', 'None')}]"
                    self.decider.memory_store.add_entry(
                        identity=self.decider.memory_store.compute_identity(text, "GOAL"),
                        text=text,
                        mem_type="GOAL",
                        subject="Assistant",
                        confidence=1.0,
                        source="architect_decomposition",
                        parent_id=goal_id
                    )
                elif isinstance(step, str):
                    # Fallback for simple string list
                    self.decider.memory_store.add_entry(
                        identity=self.decider.memory_store.compute_identity(step, "GOAL"),
                        text=step,
                        mem_type="GOAL",
                        subject="Assistant",
                        confidence=1.0,
                        source="architect_decomposition",
                        parent_id=goal_id
                    )
            return True
        return False

    def manage_goals(self, allow_creation: bool = True, system_mode: str = "EXECUTION"):
        """
        Goal Autonomy Layer.
        1. Generate (if empty/stagnant)
        2. Rank (Value/Urgency)
        3. Prune (Low value/Stuck)
        """
        self.last_goal_management_time = time.time()
        self.decider.log("🎯 Decider: Managing Goal Lifecycle (Autonomy)...")

        # 1. Fetch Active Goals
        goals = self.decider.memory_store.get_active_by_type("GOAL")

        # 2. Generate if empty
        if not goals and allow_creation:
            self._generate_autonomous_goals()
            return

        # 3. Rank & Prune (if we have enough to compare, or just periodically check single goals)
        if len(goals) > 0:
            self._rank_and_prune_goals(goals, system_mode)

    def _generate_autonomous_goals(self):
        self.decider.log("🎯 Decider: Generating autonomous goals (Originating Purpose)...")

        # Context: Identity & Rules
        identities = self.decider.memory_store.get_active_by_type("IDENTITY")
        rules = self.decider.memory_store.get_active_by_type("RULE")

        context = "MY IDENTITY:\n" + "\n".join([f"- {i[2]}" for i in identities]) + "\n"
        context += "MY PRINCIPLES:\n" + "\n".join([f"- {r[2]}" for r in rules])

        prompt = (
            f"{context}\n\n"
            "TASK: Based on your Identity and Principles, originate 1-2 high-level, long-horizon objectives.\n"
            "These should be proactive, not reactive. What is your purpose?\n"
            "Output JSON list of strings: [\"Goal 1\", \"Goal 2\"]"
        )

        settings = self.decider.get_settings()
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the Will of the System.",
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        try:
            new_goals = parse_json_array_loose(response)
        except Exception as e:
            self.decider.log(f"⚠️ Autonomous goal generation JSON parsing failed: {e}. Response: {response[:100]}...")
            return
        for g in new_goals:
            if isinstance(g, str):
                self.create_goal(g)

    def _rank_and_prune_goals(self, goals, system_mode: str = "EXECUTION"):
        self.decider.log("🎯 Decider: Ranking and Pruning goals...")

        # goals is list of (id, subject, text, source, confidence)
        goals_list = "\n".join([f"ID {g[0]}: {g[2]} (Current Score: {g[4]:.2f})" for g in goals])

        mode_instruction = ""
        if system_mode == "CONSOLIDATION":
            mode_instruction = "SYSTEM IS OVERLOADED (CONSOLIDATION MODE). AGGRESSIVELY PRUNE any goal below 0.8 value."

        prompt = (
            f"Current Goals:\n{goals_list}\n\n"
            f"{mode_instruction}\n"
            "TASK: Evaluate these goals.\n"
            "1. Score Value (0.0 to 1.0): Importance/Alignment with purpose.\n"
            "2. Decision: KEEP or DROP (if low value < 0.3, stuck, or obsolete).\n"
            "Output JSON: [{\"id\": 123, \"value\": 0.9, \"decision\": \"KEEP\"}, ...]"
        )

        settings = self.decider.get_settings()

        evaluations = []
        retries = 2

        for attempt in range(retries + 1):
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are the Strategic Planner.",
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )

            evaluations = parse_json_array_loose(response)
            if evaluations:
                break

            if attempt < retries:
                self.decider.log(f"⚠️ Failed to parse goal ranking JSON. Retrying with correction prompt ({attempt+1}/{retries})...")
                prompt += f"\n\nPREVIOUS RESPONSE WAS INVALID JSON:\n{response}\n\nFIX IT. RETURN ONLY JSON ARRAY."

        if not evaluations:
            self.decider.log("❌ Goal ranking failed after retries. Skipping pruning.")
            return

        self.decider.log(f"    📊 Goal Ranking Results ({len(evaluations)}):")

        for ev in evaluations:
            gid = ev.get("id")
            val = ev.get("value", 0.5)
            decision = ev.get("decision", "KEEP")
            self.decider.log(f"      - Goal {gid}: Score={val} -> {decision}")

            if decision == "DROP":
                self.decider.memory_store.update_type(gid, "ARCHIVED_GOAL")
                self.decider.log(f"🗑️ Decider: Pruned low-value goal {gid}")
                self.decider._track_metric("goal_abandonment", 1.0)
            else:
                self.decider.memory_store.update_confidence(gid, float(val))
