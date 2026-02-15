import unittest
from unittest.mock import MagicMock, patch
import json
import ai_core.lm
from ai_core.lm import run_local_lm, LLMError

class TestRunLocalLM(unittest.TestCase):
    def setUp(self):
        # Reset global settings for consistent testing
        self.original_settings = ai_core.lm._settings.copy()
        ai_core.lm._settings = {
            "base_url": "http://mock-url",
            "chat_model": "mock-model",
            "system_prompt": "Mock system prompt",
            "max_tokens": 100
        }

    def tearDown(self):
        ai_core.lm._settings = self.original_settings

    @patch('ai_core.lm.requests.post')
    @patch('ai_core.lm._get_epigenetics_logic', return_value="")
    def test_basic_request(self, mock_epi, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        response = run_local_lm(messages)

        self.assertEqual(response, "Hello world")
        mock_post.assert_called_once()

        # Verify payload structure
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        self.assertEqual(payload['model'], "mock-model")
        self.assertEqual(payload['messages'][0]['role'], 'system')
        self.assertEqual(payload['messages'][1]['role'], 'user')
        self.assertEqual(payload['messages'][1]['content'], 'Hi')

    @patch('ai_core.lm.requests.post')
    @patch('ai_core.lm._get_epigenetics_logic', return_value="Evolved Logic")
    def test_epigenetics_injection(self, mock_epi, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        run_local_lm(messages)

        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        system_msg = payload['messages'][0]['content']
        self.assertIn("Mock system prompt", system_msg)
        self.assertIn("[DYNAMIC EVOLVED LOGIC]", system_msg)
        self.assertIn("Evolved Logic", system_msg)

    @patch('ai_core.lm.requests.post')
    @patch('ai_core.lm._resize_and_encode_image', return_value="base64string")
    def test_vision_payload(self, mock_resize, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Image OK"}}]}
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Look at this"}]
        images = ["/path/to/image.jpg"]

        run_local_lm(messages, images=images)

        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        last_msg = payload['messages'][-1]

        self.assertEqual(last_msg['role'], 'user')
        self.assertIsInstance(last_msg['content'], list)
        self.assertEqual(last_msg['content'][0]['type'], 'text')
        self.assertEqual(last_msg['content'][0]['text'], 'Look at this')
        self.assertEqual(last_msg['content'][1]['type'], 'image_url')
        self.assertIn("base64string", last_msg['content'][1]['image_url']['url'])

    @patch('ai_core.lm.requests.post')
    def test_streaming_request(self, mock_post):
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulate lines from SSE
        chunks = [
            b'data: {"choices": [{"delta": {"content": "Stream"}}]}\n',
            b'data: {"choices": [{"delta": {"content": "ing"}}]}\n',
            b'data: [DONE]\n'
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value.__enter__.return_value = mock_response

        stop_fn = MagicMock(return_value=False)
        messages = [{"role": "user", "content": "Hi"}]

        response = run_local_lm(messages, stop_check_fn=stop_fn)

        self.assertEqual(response, "Streaming")
        self.assertTrue(stop_fn.called)

    @patch('ai_core.lm.requests.post')
    def test_context_trimming(self, mock_post):
        # Create a long history that needs trimming
        long_history = [{"role": "user", "content": "word " * 100} for _ in range(40)]

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_post.return_value = mock_response

        # Set strict max tokens to force trimming
        run_local_lm(long_history, max_tokens=1000)

        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        # The prompt + generation needs to fit in context.
        # Original logic: safe_history_tokens = 4096 - max_tokens - system_prompt - 500
        # With max_tokens=1000, we have roughly 2500 tokens for history.
        # "word " * 100 is 100 words ~ 130 tokens?
        # 20 messages * 100 words is a lot. It should be trimmed.

        self.assertLess(len(payload['messages']), len(long_history) + 1) # +1 for system prompt

if __name__ == '__main__':
    unittest.main()
