import re
import json
from enum import Enum
from typing import TYPE_CHECKING, List, Dict, Any, Optional
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
                self._perform_linear_chain(topic)
            elif selected_strategy == ThinkingStrategy.DIALECTIC:
                self._perform_dialectical(topic)
            elif selected_strategy == ThinkingStrategy.FIRST_PRINCIPLES:
                self._perform_first_principles(topic)
            else:
                # Default to Tree of Thoughts
                self._perform_tree_of_thoughts(topic, max_depth, beam_width)
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

    def _perform_linear_chain(self, topic: str):
        """Simple sequential reasoning."""
        context = self._gather_thinking_context(topic)
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
            system_prompt="You are a Logical Engine.",
            max_tokens=800,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model"),
            stop_check_fn=self.decider.stop_check
        )

        self.decider.log(f"🧠 Linear Reasoning Complete.")
        self.decider.command_executor.create_note(f"Linear Reasoning ({topic}):\n{chain}")
        self.decider.reasoning_store.add(content=f"Linear Reasoning ({topic}): {chain}", source="linear_reasoning", confidence=1.0)
        self.decider.decision_maker._reflect_on_decision(f"Linear Reasoning on {topic}", chain[:200] + "...")


    def _perform_dialectical(self, topic: str):
        """Dialectical debate (Thesis-Antithesis-Synthesis)."""
        if self.decider.dialectics:
            context = self._gather_thinking_context(topic)
            result = self.decider.dialectics.run_debate(topic, context=context, reasoning_store=self.decider.reasoning_store)
            self.decider.command_executor.create_note(f"Dialectic Synthesis ({topic}):\n{result}")
        else:
            self.decider.log("⚠️ Dialectics component missing. Fallback to Linear.")
            self._perform_linear_chain(topic)

    def _perform_first_principles(self, topic: str):
        """First Principles decomposition."""
        context = self._gather_thinking_context(topic)

        # Step 1: Deconstruct
        deconstruct_prompt = (
            f"{context}\n"
            "TASK: Deconstruct this topic into its most fundamental truths or axioms.\n"
            "Discard all assumptions, analogies, and social conventions.\n"
            "List ONLY the undeniable facts."
        )
        axioms = run_local_lm(
            messages=[{"role": "user", "content": deconstruct_prompt}],
            system_prompt="You are a First Principles Thinker.",
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
            system_prompt="You are a First Principles Thinker.",
            max_tokens=600,
            base_url=self.decider.get_settings().get("base_url"),
            chat_model=self.decider.get_settings().get("chat_model")
        )

        full_text = f"First Principles Analysis on {topic}:\n\nAXIOMS:\n{axioms}\n\nCONCLUSION:\n{conclusion}"
        self.decider.command_executor.create_note(full_text)
        self.decider.reasoning_store.add(content=full_text, source="first_principles", confidence=1.0)
        self.decider.decision_maker._reflect_on_decision(f"First Principles on {topic}", conclusion[:200] + "...")

    def _perform_tree_of_thoughts(self, topic: str, max_depth: int = 10, beam_width: int = 3):
        """Execute a Tree of Thoughts (ToT) reasoning process."""
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

        for path in active_paths:
            if self.decider.stop_check(): break

            # Generate N possible next steps for this path
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

            # Evaluation Phase (Scoring)
            for step in next_steps:
                if not isinstance(step, str): continue

                # Check for conclusion
                if "[CONCLUSION]" in step:
                    final_conclusion = step.replace("[CONCLUSION]", "").strip()
                    best_path = path + [step]
                    break

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

                candidates.append((path + [step], score))

            if final_conclusion:
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
