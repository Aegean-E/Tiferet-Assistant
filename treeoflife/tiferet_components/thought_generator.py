import re
import json
from typing import TYPE_CHECKING, List, Dict, Any
from ai_core.lm import compute_embedding, run_local_lm
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class ThoughtGenerator:
    def __init__(self, decider: 'Decider'):
        self.decider = decider

    def perform_thinking_chain(self, topic: str, max_depth: int = 10, beam_width: int = 3):
        """Execute a Tree of Thoughts (ToT) reasoning process."""
        self.decider.log(f"🌳 Decider starting Tree of Thoughts on: {topic}")
        if self.decider.chat_fn:
            self.decider.chat_fn("Decider", f"🌳 Starting Tree of Thoughts: {topic}")

        # 1. Gather Context
        static_context = self._gather_thinking_context(topic)

        # Tree State: List of paths. A path is a list of thought strings.
        # Start with one empty path
        active_paths = [[]]
        final_conclusion = None
        best_path = []

        for depth in range(1, max_depth + 1):
            if self.decider.stop_check():
                break

            if final_conclusion:
                break

            self.decider.log(f"🌳 Depth {depth}: Expanding {len(active_paths)} paths...")

            # Expansion Phase (Branching & Evaluation)
            candidates, found_conclusion, found_path = self._expand_thought_paths(active_paths, static_context, beam_width, topic)

            if found_conclusion:
                final_conclusion = found_conclusion
                best_path = found_path
                break

            # Selection Phase (Beam Search)
            active_paths = self._select_best_paths(candidates, beam_width, depth, topic)
            if active_paths:
                # Update best path to the current best just in case we stop prematurely
                best_path = active_paths[0]

        if not final_conclusion and active_paths:
            # If max depth reached without conclusion, take the best path
            best_path = active_paths[0]
            final_conclusion = "Max depth reached. Best partial reasoning path selected."
        elif final_conclusion:
             self.decider.command_executor.create_note(f"Conclusion on {topic}: {final_conclusion}")

        # Post-chain Summarization
        if best_path:
            summary = self._summarize_thinking_chain(topic, best_path)

            # Save summary as note if no formal conclusion was reached (e.g. interrupted by loop)
            # This ensures partial progress is preserved in memory
            if not final_conclusion:
                self.decider.command_executor.create_note(f"ToT Summary ({topic}): {summary}")

        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("wait", 0, "Thinking chain complete")

    def _gather_thinking_context(self, topic: str) -> str:
        """Helper to gather relevant memories, documents, and subjective context for ToT."""
        settings = self.decider.get_settings()
        subjective_context = ""

        # 0.5 Retrieve Subjective Memories
        if hasattr(self.decider.meta_memory_store, 'search'):
            query_embedding = compute_embedding(topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
            subj_results = self.decider.meta_memory_store.search(query_embedding, limit=3)
            if subj_results:
                subjective_context = "Relevant Subjective Experiences (How I felt before):\n" + "\n".join([f"- [{m['event_type']}] {m['text']}" for m in subj_results]) + "\n"

        # 1. Gather Context (Memories & Docs)
        query_embedding = compute_embedding(
            topic,
            base_url=settings.get("base_url"),
            embedding_model=settings.get("embedding_model")
        )

        mem_results = self.decider.memory_store.search(query_embedding, limit=5)
        doc_results = self.decider.document_store.search_chunks(query_embedding, top_k=3)

        context_str = ""
        if mem_results:
            context_str += "Relevant Memories:\n" + "\n".join([f"- {m[3]}" for m in mem_results]) + "\n"
        if doc_results:
            context_str += "Relevant Documents:\n" + "\n".join([f"- {d['text'][:300]}..." for d in doc_results]) + "\n"

        static_context = f"Topic: {topic}\n"
        if context_str:
            static_context += f"\n{context_str}\n"
        if subjective_context:
            static_context += f"\n{subjective_context}\n"

        # Ask Daat for structure
        if self.decider.daat:
            structure = self.decider.daat.provide_reasoning_structure(topic)
            if structure and not structure.startswith("⚠️"):
                static_context += f"\nReasoning Structure (Guide):\n{structure}\n"

        return static_context

    def _expand_thought_paths(self, active_paths: List[List[str]], static_context: str, beam_width: int, topic: str):
        """Helper to expand current thought paths using LLM."""
        candidates = []
        final_conclusion = None
        best_path = []
        settings = self.decider.get_settings()

        def expand_single_path(path):
            if self.decider.stop_check(): return None

            path_str = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(path)])
            prompt = (
                f"You are an AGI reasoning engine using Tree of Thoughts.\n"
                f"{static_context}\n"
                f"Current Path:\n{path_str}\n\n"
                f"Generate {beam_width} distinct, valid next steps to advance reasoning towards a solution.\n"
                "CRITICAL REVIEW: Ensure steps are logical, novel, and directly address the topic.\n"
                "If a solution is reached, start the step with '[CONCLUSION]'.\n"
                "Output JSON list of strings: [\"Step A...\", \"Step B...\", \"Step C...\"]"
            )

            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Generator.",
                temperature=0.7,
                max_tokens=400,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.decider.stop_check
            )

            next_steps = parse_json_array_loose(response)
            if not next_steps:
                next_steps = [response.strip()]

            path_candidates = []
            local_conclusion = None
            local_best_path = []

            for step in next_steps:
                if not isinstance(step, str): continue

                # Check for conclusion
                if "[CONCLUSION]" in step:
                    local_conclusion = step.replace("[CONCLUSION]", "").strip()
                    local_best_path = path + [step]
                    # Early exit for this path
                    return [(path + [step], 1.0)], local_conclusion, local_best_path

                # Score the step
                eval_prompt = (
                    f"Evaluate this reasoning step for the topic '{topic}':\n"
                    f"Step: \"{step}\"\n"
                    "Criteria: Logic, Relevance, Novelty.\n"
                    "Output a score from 0.0 to 1.0."
                )

                score_str = run_local_lm(
                    messages=[{"role": "user", "content": eval_prompt}],
                    system_prompt="You are an Evaluator.",
                    temperature=0.1,
                    max_tokens=10,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model")
                )

                try:
                    score = float(re.search(r"0\.\d+|1\.0|0|1", score_str).group())
                except:
                    score = 0.5

                path_candidates.append((path + [step], score))

            return path_candidates, local_conclusion, local_best_path

        # Execute in parallel if executor is available
        if self.decider.executor and len(active_paths) > 1:
            futures = [self.decider.executor.submit(expand_single_path, path) for path in active_paths]
            for future in futures:
                try:
                    res = future.result()
                    if res:
                        p_candidates, p_conclusion, p_best_path = res
                        candidates.extend(p_candidates)
                        if p_conclusion:
                            final_conclusion = p_conclusion
                            best_path = p_best_path
                            # If we found a conclusion, we can potentially stop early,
                            # but let's gather all results to be safe or break?
                            # For simplicity, we keep gathering but final_conclusion is set.
                except Exception as e:
                    self.decider.log(f"⚠️ Error expanding path: {e}")
        else:
            # Serial execution
            for path in active_paths:
                res = expand_single_path(path)
                if res:
                    p_candidates, p_conclusion, p_best_path = res
                    candidates.extend(p_candidates)
                    if p_conclusion:
                        final_conclusion = p_conclusion
                        best_path = p_best_path
                        break

        return candidates, final_conclusion, best_path

    def _select_best_paths(self, candidates: List[tuple], beam_width: int, depth: int, topic: str) -> List[List[str]]:
        """Helper to select top K paths using Beam Search."""
        candidates.sort(key=lambda x: x[1], reverse=True)
        active_paths = [c[0] for c in candidates[:beam_width]]

        # Log best step of this depth
        if active_paths:
            best_step = active_paths[0][-1]
            self.decider.log(f"🌳 Best Step at Depth {depth}: {best_step[:100]}...")
            if self.decider.chat_fn:
                self.decider.chat_fn("Decider", f"🌳 Depth {depth}: {best_step}")

            # Store in reasoning
            self.decider.reasoning_store.add(content=f"ToT Depth {depth} ({topic}): {best_step}", source="decider_tot", confidence=1.0)

        return active_paths

    def _summarize_thinking_chain(self, topic: str, best_path: List[str]) -> str:
        """Helper to summarize the reasoning chain."""
        self.decider.log(f"🧠 Generating summary of Tree of Thoughts...")
        settings = self.decider.get_settings()
        full_chain_text = "\n".join(best_path)
        summary_prompt = (
            f"Synthesize the following reasoning path regarding '{topic}' into a clear, comprehensive summary for the user.\n"
            f"Include key insights and the final conclusion if reached.\n\n"
            f"Reasoning Path:\n{full_chain_text}"
        )

        summary = run_local_lm(
            messages=[{"role": "user", "content": summary_prompt}],
            system_prompt="You are a helpful assistant summarizing your internal reasoning.",
            temperature=0.5,
            max_tokens=500,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model"),
            stop_check_fn=self.decider.stop_check
        )

        if self.decider.chat_fn:
            self.decider.chat_fn("Decider", f"🌳 Tree of Thoughts Summary:\n{summary}")

        # Metacognitive Reflection on the thinking process
        self.decider.decision_maker._reflect_on_decision(f"Tree of Thoughts on {topic}", summary)

        return summary

    def perform_self_reflection(self):
        """
        Deep introspection cycle.
        Analyzes alignment between Identity, Goals, and recent Actions.
        """
        self.decider.log("🧘 Decider entering Self-Reflection...")

        # 1. Gather Self-Context
        identity = self.decider.memory_store.get_active_by_type("IDENTITY")
        goals = self.decider.memory_store.get_active_by_type("GOAL")
        recent_actions = self.decider.meta_memory_store.list_recent(limit=10)

        context = "MY IDENTITY:\n" + "\n".join([f"- {i[2]}" for i in identity[:5]]) + "\n\n"
        context += "MY GOALS:\n" + "\n".join([f"- {g[2]}" for g in goals[:5]]) + "\n\n"
        context += "RECENT ACTIONS:\n" + "\n".join([f"- {a[3]}" for a in recent_actions])

        # 2. Reflect
        prompt = (
            f"{context}\n\n"
            "TASK: Reflect on your recent actions. Are they aligned with your Identity and Goals?\n"
            "Identify 1 area of improvement or 1 new insight about yourself.\n"
            "Output a concise reflection."
        )

        reflection = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Self-Aware AI.",
            max_tokens=300,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        )

        self.decider.log(f"🧘 Reflection: {reflection}")

        # 3. Store
        self.decider.reasoning_store.add(content=f"Self-Reflection: {reflection}", source="decider_reflection", confidence=1.0)

        # Metacognitive Reflection on the reflection
        self.decider.decision_maker._reflect_on_decision("Deep Self-Reflection", reflection)
