import sys
import os
import threading
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.append(os.getcwd())

# Mock dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['numpy'].__version__ = "1.24.0"
sys.modules['faiss'] = MagicMock()
sys.modules['openai_whisper'] = MagicMock()
sys.modules['whisper'] = MagicMock()
sys.modules['pyvis'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['dowhy'] = MagicMock()
sys.modules['ttkbootstrap'] = MagicMock()
sys.modules['PyMuPDF'] = MagicMock()
sys.modules['fitz'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# 1. Verify GPU_SEMAPHORE update
import ai_core.lm
print(f"Checking GPU_SEMAPHORE value... {ai_core.lm.GPU_SEMAPHORE._value}")
assert ai_core.lm.GPU_SEMAPHORE._value == 4, "GPU_SEMAPHORE was not updated to 4"
print("✅ GPU_SEMAPHORE verification passed.")

# 2. Verify ThoughtGenerator new method and Daat integration
from treeoflife.tiferet_components.thought_generator import ThoughtGenerator

# Mock Decider and Daat
mock_decider = MagicMock()
mock_decider.get_settings.return_value = {"base_url": "mock", "chat_model": "mock", "embedding_model": "mock-embed"}
mock_decider.stop_check.return_value = False
mock_decider.daat = MagicMock()
mock_decider.daat.spreading_activation_search.return_value = [{"text": "Deep Context"}]
mock_decider.daat.provide_reasoning_structure.return_value = "Structure"

# Patch compute_embedding to return dummy vector
with patch('treeoflife.tiferet_components.thought_generator.compute_embedding', return_value=[0.1, 0.2]):
    tg = ThoughtGenerator(mock_decider)

    # Check context gathering
    context = tg._gather_thinking_context("Test Topic")
    print(f"Context gathered: {context[:50]}...")
    assert "Deep Semantic Associations" in context, "Daat context not integrated"
    print("✅ Daat context integration passed.")

    # Check evolve_stream_of_consciousness exists
    assert hasattr(tg, 'evolve_stream_of_consciousness'), "evolve_stream_of_consciousness missing"
    print("✅ evolve_stream_of_consciousness exists.")

    # Check if run_local_lm calls are parallelized (by checking imports basically)
    from concurrent.futures import ThreadPoolExecutor
    assert ThreadPoolExecutor in tg._expand_thought_paths.__globals__.values(), "ThreadPoolExecutor not imported in thought_generator"
    print("✅ Parallelism imports verified.")

# 3. Verify Phenomenology Cleanup
from ai_core.core_phenomenology import Phenomenology
assert not hasattr(Phenomenology, '_generate_internal_monologue'), "_generate_internal_monologue should be removed"
print("✅ Phenomenology cleanup passed.")

# 4. Verify GlobalWorkspace Cleanup
from ai_core.core_spotlight import GlobalWorkspace
assert not hasattr(GlobalWorkspace, 'evolve_thought'), "evolve_thought should be removed from GlobalWorkspace"
print("✅ GlobalWorkspace cleanup passed.")

print("🚀 All verifications passed successfully.")
