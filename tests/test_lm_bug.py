import unittest
from unittest.mock import MagicMock, patch
import ai_core.lm
from ai_core.lm import run_local_lm, LLMError

class TestLMBug(unittest.TestCase):
    def setUp(self):
        self.original_settings = ai_core.lm._settings.copy()
        ai_core.lm._settings = {
            "base_url": "http://mock-url",
            "chat_model": "mock-model",
            "system_prompt": "Base Prompt",
            "max_tokens": 100
        }

    def tearDown(self):
        ai_core.lm._settings = self.original_settings

    @patch('ai_core.lm.requests.post')
    @patch('ai_core.lm._get_epigenetics_logic', return_value="EVOLVED LOGIC")
    def test_duplicate_injection_on_retry(self, mock_epi, mock_post):
        # First call fails with 400 (Bad Request - Context Length), triggering retry
        # Second call succeeds

        response1 = MagicMock()
        response1.status_code = 400
        response1.raise_for_status.side_effect = Exception("400 Bad Request")

        response2 = MagicMock()
        response2.status_code = 200
        response2.json.return_value = {"choices": [{"message": {"content": "Success"}}]}

        mock_post.side_effect = [Exception("400 Bad Request"), response2]

        messages = [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}]

        run_local_lm(messages)

        self.assertEqual(mock_post.call_count, 2)

        # Check the payload of the second call
        args, kwargs = mock_post.call_args_list[1]
        payload = kwargs['json']
        system_msg = payload['messages'][0]['content']

        print(f"DEBUG: System Prompt in Retry:\n{system_msg}")

        # Count occurrences of logic marker
        count = system_msg.count("[DYNAMIC EVOLVED LOGIC]")
        self.assertEqual(count, 1, "Logic marker should appear exactly once (Fix verification)")

if __name__ == '__main__':
    unittest.main()
