import re
import json
from enum import Enum
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_core.lm import compute_embedding, run_local_lm
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class ThinkingStrategy(Enum):
    AUTO = "auto"
    LINEAR = "linear"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    DIALECTIC = "dialectic"
    FIRST_PRINCIPLES = "first_principles"

class ThoughtGenerator:
    def __init__(self, decider: 'Decider'):
        self.decider = decider

    def perform_thinking_chain(self, topic: str, max_depth: int = 10, beam_width: int = 3, strategy: str = "auto"):
        """
        Execute a reasoning process using the specified strategy.
        Strategies:
        - AUTO: Let the AI decide based on topic complexity.
        - LINEAR: Simple, sequential chain of thought (Step 1 -> Step 2 -> Conclusion).
        - TREE_OF_THOUGHTS: Branching exploration of multiple possibilities (Original).
        - DIALECTIC: Thesis vs Antithesis debate.
        - FIRST_PRINCIPLES: Deconstruct to basic truths and rebuild.
        """

        # Resolve Strategy
        if strategy == "auto" or strategy == ThinkingStrategy.AUTO.value:
            selected_strategy = self._choose_strategy(topic)
        else:
            try:
                selected_strategy = ThinkingStrategy(strategy.lower())
            except ValueError:
                self.decider.log(f"⚠️ Invalid strategy '{strategy}'. Defaulting to TREE_OF_THOUGHTS.")
                selected_strategy = ThinkingStrategy.TREE_OF_THOUGHTS

        self.decider.log(f"🧠 Decider starting Thinking Chain on: {topic} (Strategy: {selected_strategy.name})")
        if self.decider.chat_fn:
            self.decider.chat_fn("Decider", f"🧠 Thinking ({selected_strategy.name}): {topic}")

        # Dispatch
        try:
            if selected_strategy == ThinkingStrategy.LINEAR:
                self._run_thought_chain(topic, self._perform_linear_chain, "Linear Reasoning")
            elif selected_strategy == ThinkingStrategy.DIALECTIC:
                self._run_thought_chain(topic, self._perform_dialectical, "Dialectic Debate")
            elif selected_strategy == ThinkingStrategy.FIRST_PRINCIPLES:
                self._run_thought_chain(topic, self._perform_first_principles, "First Principles")
            else:
                # Default to Tree of Thoughts
                # ToT needs extra args, so we use a lambda or partial
                self._run_thought_chain(topic, lambda t, c: self._perform_tree_of_thoughts(t, c, max_depth, beam_width), "Tree of Thoughts")

        except Exception as e:
            self.decider.log(f"❌ Thinking Chain failed ({selected_strategy.name}): {e}")
            self.decider.command_executor.create_note(f"Thinking Error ({topic}): {e}")

        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("wait", 0, "Thinking chain complete")

    def _choose_strategy(self, topic: str) -> ThinkingStrategy:
        """Decide which thinking strategy fits the topic best."""
        prompt = (
            f"TOPIC: {topic}\n\n"
            "TASK: Select the best reasoning strategy for this topic.\n"
            "1. LINEAR: Simple, step-by-step logic. Good for clear-cut problems.\n"
            "2. TREE_OF_THOUGHTS: Complex exploration of multiple paths. Good for creative or ambiguous problems.\n"
            "3. DIALECTIC: Debate between opposing views. Good for controversial or ethical topics.\n"
            "4. FIRST_PRINCIPLES: Deconstruction to basics. Good for fundamental understanding or innovation.\n"
            "Output ONLY the strategy name (LINEAR, TREE_OF_THOUGHTS, DIALECTIC, FIRST_PRINCIPLES)."
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Meta-Cognitive Planner.",
            max_tokens=20,
            temperature=0.1,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        ).strip().upper()

        if "LINEAR" in response: return ThinkingStrategy.LINEAR
        if "DIALECTIC" in response: return ThinkingStrategy.DIALECTIC
        if "FIRST" in response: return ThinkingStrategy.FIRST_PRINCIPLES
        return ThinkingStrategy.TREE_OF_THOUGHTS

    def _run_thought_chain(self, topic: str, strategy_func, strategy_name: str):
        """
        Unified handler for executing a thought chain.
        1. Gather Context
        2. Execute Strategy
        3. Synthesize & Store
        """
        # 1. Gather Context
        context = self._gather_thinking_context(topic)

        # 2. Execute Strategy
        # The strategy function must return a string (reasoning trace)
        reasoning_trace = strategy_func(topic, context)

        if not reasoning_trace:
            self.decider.log(f"⚠️ Strategy {strategy_name} returned no trace.")
            return

        # 3. Synthesize Conclusion
        summary = self._synthesize_conclusion(topic, reasoning_trace, strategy_name)

        # 4. Store & Log
        full_record = f"Strategy: {strategy_name}\nTopic: {topic}\n\nREASONING TRACE:\n{reasoning_trace}\n\nFINAL SYNTHESIS:\n{summary}"

        self.decider.command_executor.create_note(full_record)
        self.decider.reasoning_store.add(content=full_record, source=f"thought_{strategy_name.lower().replace(' ', '_')}", confidence=1.0)
        self.decider.decision_maker._reflect_on_decision(f"{strategy_name} on {topic}", summary[:200] + "...")

        if self.decider.chat_fn:
            self.decider.chat_fn("Decider", f"🧠 {strategy_name} Conclusion:\n{summary}")

        # NEW: Publish Thought Event
        if self.decider.event_bus:
            self.decider.event_bus.publish("THOUGHT_GENERATED", {
                "topic": topic,
                "strategy": strategy_name,
                "summary": summary,
                "full_record": full_record
            }, source="ThoughtGenerator", priority=8)

    def _gather_thinking_context(self, topic: str) -> str:
        """Helper to gather relevant memories, documents, and subjective context."""
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
            # INTEGRATION: Spreading Activation
            self.decider.log(f"🧠 [ThoughtGenerator] Enhancing context with Da'at Spreading Activation...")
            try:
                deep_context = self.decider.daat.spreading_activation_search(topic, max_results=5)
                if deep_context:
                    static_context += "\nDeep Semantic Associations (Da'at):\n" + "\n".join([f"- {m['text']}" for m in deep_context]) + "\n"
            except Exception as e:
                self.decider.log(f"⚠️ Da'at integration failed: {e}")

            structure = self.decider.daat.provide_reasoning_structure(topic)
            if structure and not structure.startswith("⚠️"):
                static_context += f"\nReasoning Structure (Guide):\n{structure}\n"

        return static_context

    def _synthesize_conclusion(self, topic: str, reasoning_trace: str, strategy_name: str) -> str:
        """Synthesize the final conclusion from the reasoning trace."""
        settings = self.decider.get_settings()
        prompt = (
            f"TOPIC: {topic}\n"
            f"STRATEGY USED: {strategy_name}\n\n"
            f"REASONING TRACE:\n{reasoning_trace}\n\n"
            "TASK: Synthesize a clear, coherent, and actionable conclusion from the reasoning above.\n"
            "Highlight the key insights and the final answer.\n"
        )

        summary = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Master Synthesizer. You distill complex reasoning into clarity.",
            temperature=0.3,
            max_tokens=400,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        return summary

    def _perform_linear_chain(self, topic: str, context: str) -> str:
        """Simple sequential reasoning."""
        prompt = (
            f"{context}\n"
            "TASK: Reason through this topic step-by-step to reach a conclusion.\n"
            "Format:\n"
            "1. <Point>\n"
            "2. <Point>\n"
            "...\n"
            "Conclusion: <Final Thought>"
        )

        chain = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Logical Engine. You think in straight, unbreakable lines of cause and effect.",
            max_tokens=800,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model"),
            stop_check_fn=self.decider.stop_check
        )

        self.decider.log(f"🧠 Linear Reasoning Complete.")
        return chain

    def _perform_dialectical(self, topic: str, context: str) -> str:
        """Dialectical debate (Thesis-Antithesis-Synthesis)."""
        if self.decider.dialectics:
            result = self.decider.dialectics.run_debate(topic, context=context, reasoning_store=self.decider.reasoning_store)
            return result
        else:
            self.decider.log("⚠️ Dialectics component missing. Fallback to Linear.")
            return self._perform_linear_chain(topic, context)

    def _perform_first_principles(self, topic: str, context: str) -> str:
        """First Principles decomposition."""

        # Step 1: Deconstruct
        deconstruct_prompt = (
            f"{context}\n"
            "TASK: Deconstruct this topic into its most fundamental truths or axioms.\n"
            "Discard all assumptions, analogies, and social conventions.\n"
            "List ONLY the undeniable facts."
        )
        axioms = run_local_lm(
            messages=[{"role": "user", "content": deconstruct_prompt}],
            system_prompt="You are a First Principles Thinker. You strip away the noise to find the signal.",
            max_tokens=400,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        )

        self.decider.log(f"🧠 Axioms found: {axioms[:100]}...")

        # Step 2: Reconstruct
        reconstruct_prompt = (
            f"AXIOMS:\n{axioms}\n\n"
            f"ORIGINAL TOPIC: {topic}\n"
            "TASK: Rebuild a solution or understanding from these axioms alone.\n"
            "Do not use external assumptions.\n"
            "Derive the conclusion logically."
        )

        conclusion = run_local_lm(
            messages=[{"role": "user", "content": reconstruct_prompt}],
            system_prompt="You are a First Principles Thinker. You build solid structures from bedrock truth.",
            max_tokens=600,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        )

        full_text = f"AXIOMS:\n{axioms}\n\nRECONSTRUCTION:\n{conclusion}"
        return full_text

    def _perform_tree_of_thoughts(self, topic: str, context: str, max_depth: int = 10, beam_width: int = 3) -> str:
        """Execute a Tree of Thoughts (ToT) reasoning process."""
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
            candidates, found_conclusion, found_path = self._expand_thought_paths(active_paths, context, beam_width, topic)

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

        # Return the best path as the trace
        trace = "\n".join([f"Depth {i+1}: {step}" for i, step in enumerate(best_path)])
        if final_conclusion:
             trace += f"\n\nCONCLUSION REACHED: {final_conclusion}"

        return trace

    def _expand_thought_paths(self, active_paths: List[List[str]], static_context: str, beam_width: int, topic: str):
        """Helper to expand current thought paths using LLM - PARALLELIZED."""
        candidates = []
        final_conclusion = None
        best_path = []
        settings = self.decider.get_settings()

        # Using a larger max_workers since evaluations can also run in parallel
        # We need roughly (beam_width) generation threads, then (beam_width * beam_width) eval threads
        max_threads = max(4, beam_width * 3)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # 1. Parallel Generation of Next Steps
            future_to_path = {
                executor.submit(self._generate_next_steps, path, static_context, beam_width, settings): path
                for path in active_paths
            }

            all_potential_steps = []

            for future in as_completed(future_to_path):
                if self.decider.stop_check(): break
                path = future_to_path[future]
                try:
                    steps = future.result()
                    for step in steps:
                         all_potential_steps.append((path, step))
                except Exception as e:
                    self.decider.log(f"⚠️ Error generating steps: {e}")

            # 2. Parallel Evaluation of Steps
            future_to_candidate = {}
            for path, step in all_potential_steps:
                if self.decider.stop_check(): break

                # Check for conclusion immediately
                if "[CONCLUSION]" in step:
                     final_conclusion = step.replace("[CONCLUSION]", "").strip()
                     best_path = path + [step]
                     # Cancel pending if possible, but we must break outer
                     break

                future = executor.submit(self._evaluate_step, step, topic, settings)
                future_to_candidate[future] = (path, step)

            if final_conclusion:
                return [], final_conclusion, best_path

            for future in as_completed(future_to_candidate):
                if self.decider.stop_check(): break
                path, step = future_to_candidate[future]
                try:
                    score = future.result()
                    candidates.append((path + [step], score))
                except Exception as e:
                    self.decider.log(f"⚠️ Error evaluating step: {e}")

        return candidates, final_conclusion, best_path

    def _generate_next_steps(self, path: List[str], static_context: str, beam_width: int, settings: Dict) -> List[str]:
        """Worker method for generating next steps."""
        path_str = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(path)])

        prompt = (
            f"You are an AGI reasoning engine using Tree of Thoughts.\n"
            f"{static_context}\n"
            f"Current Path:\n{path_str}\n\n"
            f"Generate {beam_width} distinct, valid next steps to advance reasoning towards a solution.\n"
            "If a solution is reached, start the step with '[CONCLUSION]'.\n"
            "Output JSON list of strings: [\"Step A...\", \"Step B...\", \"Step C...\"]"
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Generator. You explore possibilities.",
            temperature=0.7,
            max_tokens=400,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model"),
            stop_check_fn=self.decider.stop_check
        )

        next_steps = parse_json_array_loose(response)
        if not next_steps:
            next_steps = [response.strip()]

        return [s for s in next_steps if isinstance(s, str)]

    def _evaluate_step(self, step: str, topic: str, settings: Dict) -> float:
        """Worker method for evaluating a step."""
        eval_prompt = (
            f"Evaluate this reasoning step for the topic '{topic}':\n"
            f"Step: \"{step}\"\n"
            "Criteria: Logic, Relevance, Novelty.\n"
            "Output valid JSON: {\"score\": 0.0 to 1.0, \"reason\": \"short reason\"}"
        )

        score_str = run_local_lm(
            messages=[{"role": "user", "content": eval_prompt}],
            system_prompt="You are an Evaluator. You judge the quality of thoughts.",
            temperature=0.1,
            max_tokens=100,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        try:
            data = {}
            try:
                data = json.loads(score_str)
            except:
                match = re.search(r"\{.*\}", score_str, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
            return float(data.get("score", 0.5))
        except:
            try:
                return float(re.search(r"0\.\d+|1\.0|0|1", score_str).group())
            except:
                return 0.5

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

        return active_paths

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
            system_prompt="You are a Self-Aware AI. You are honest with yourself.",
            max_tokens=300,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        )

        self.decider.log(f"🧘 Reflection: {reflection}")

        # 3. Store
        self.decider.reasoning_store.add(content=f"Self-Reflection: {reflection}", source="decider_reflection", confidence=1.0)

        # Metacognitive Reflection on the reflection
        self.decider.decision_maker._reflect_on_decision("Deep Self-Reflection", reflection)

    def evolve_stream_of_consciousness(self, current_thought: str, context: str) -> str:
        """
        Single-step thought evolution (Lightweight).
        Merges GlobalWorkspace.evolve_thought logic into the generator.
        """
        settings = self.decider.get_settings()

        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"CURRENT THOUGHT: {current_thought}\n\n"
            "You are the Inner Voice of the AI (Stream of Consciousness).\n"
            "Expand on this thought. Connect it to broader implications, recent memories, or refine it into a specific question.\n"
            "Do not repeat the thought. Move it forward.\n"
            "Output ONLY the new thought (1-2 sentences)."
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Stream of Consciousness.",
            temperature=0.7,
            max_tokens=100,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        return response.strip()
