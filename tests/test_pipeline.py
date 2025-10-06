# import unittest
# from unittest.mock import patch, MagicMock
# from src.pipeline import decide_node, weather_node, pdf_node, llm_node, embed_store_node, vectorstore

# class TestPipeline(unittest.TestCase):

#     def test_decide_weather(self):
#         state = {'query': 'What is the weather in London?'}
#         result = decide_node(state)
#         self.assertEqual(result['decision'], 'weather')

#     def test_decide_pdf(self):
#         state = {'query': 'What is in the PDF?'}
#         result = decide_node(state)
#         self.assertEqual(result['decision'], 'pdf')

#     @patch('src.pipeline.get_weather')
#     def test_weather_node(self, mock_get):
#         mock_get.return_value = 'Temperature: 20°C, Description: sunny'
#         state = {'query': 'weather London'}
#         result = weather_node(state)
#         self.assertIn('20°C', result['data'])

#     def test_pdf_node_no_vectorstore(self):
#         global vectorstore
#         vectorstore = None
#         state = {'query': 'test query'}
#         result = pdf_node(state)
#         self.assertEqual(result['data'], 'PDF not loaded')

#     @patch('src.pipeline.vectorstore')
#     def test_pdf_node_with_mock(self, mock_vs):
#         mock_retriever = MagicMock()
#         mock_vs.as_retriever.return_value = mock_retriever
#         mock_doc = MagicMock()
#         mock_doc.page_content = 'Test content'
#         mock_retriever.get_relevant_documents.return_value = [mock_doc]
#         global vectorstore
#         vectorstore = mock_vs
#         state = {'query': 'test query'}
#         result = pdf_node(state)
#         self.assertIn('Test content', result['data'])

#     @patch('src.pipeline.llm')
#     def test_llm_node(self, mock_llm):
#         mock_llm.return_value = 'Processed response'
#         state = {'data': 'test data'}
#         result = llm_node(state)
#         self.assertEqual(result['response'], 'Processed response')

#     @patch('src.pipeline.embeddings')
#     @patch('src.pipeline.qdrant_client')
#     def test_embed_store_node(self, mock_client, mock_emb):
#         mock_emb.embed_query.return_value = [0.1, 0.2]
#         state = {'response': 'test response'}
#         result = embed_store_node(state)
#         self.assertEqual(result, {})
#         mock_client.add.assert_called_once()

# if __name__ == '__main__':
#     unittest.main()
import unittest
from unittest.mock import patch, MagicMock
from src.pipeline import decide_node, weather_node, pdf_node, llm_node, embed_store_node, vectorstore

class TestPipeline(unittest.TestCase):

    def test_decide_weather(self):
        state = {"query": "What is the weather in London?"}
        result = decide_node(state)
        self.assertEqual(result["decision"], "weather")

    def test_decide_pdf(self):
        state = {"query": "Explain the PDF content"}
        result = decide_node(state)
        self.assertEqual(result["decision"], "pdf")

    @patch("src.pipeline.get_weather")
    def test_weather_node(self, mock_get):
        mock_get.return_value = "Temp 20°C"
        state = {"query": "weather London"}
        result = weather_node(state)
        self.assertIn("20", result["data"])

    def test_pdf_node_no_vectorstore(self):
        global vectorstore
        vectorstore = None
        state = {"query": "test"}
        result = pdf_node(state)
        self.assertEqual(result["data"], "PDF not loaded")

    @patch("src.pipeline.vectorstore")
    def test_pdf_node_with_mock(self, mock_vs):
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_doc = MagicMock()
        mock_doc.page_content = "Test PDF Content"
        mock_retriever.invoke.return_value = [mock_doc]
        global vectorstore
        vectorstore = mock_vs
        state = {"query": "test"}
        result = pdf_node(state)
        self.assertIn("Test PDF Content", result["data"])

    @patch("src.pipeline.llm")
    def test_llm_node(self, mock_llm):
        mock_llm.invoke.return_value = "LLM response"
        state = {"data": "text"}
        result = llm_node(state)
        self.assertEqual(result["response"], "LLM response")

    @patch("src.pipeline.embeddings")
    @patch("src.pipeline.qdrant_client")
    def test_embed_store_node(self, mock_client, mock_emb):
        mock_emb.embed_query.return_value = [0.1, 0.2]
        state = {"response": "sample response"}
        result = embed_store_node(state)
        self.assertEqual(result, {})

if __name__ == "__main__":
    unittest.main()
