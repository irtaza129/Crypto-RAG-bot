# Basic test for chatbot API
import unittest
from app import app

class ChatbotTestCase(unittest.TestCase):
	def setUp(self):
		self.app = app.test_client()

	def test_chat_endpoint(self):
		response = self.app.post('/chat', json={"query": "What is crypto compliance?"})
		self.assertEqual(response.status_code, 200)
		self.assertIn("answer", response.get_json())

if __name__ == "__main__":
	unittest.main()
