"""
模块② 文献检索 Agent — 单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from config import CONFIG
from agents.literature_agent import LiteratureAgent, LightKnowledgeGraph
from tools.arxiv_search import ArXivSearchTool, Paper
from tools.llm_client import LLMClient
from memory.memory_store import MemoryStore
from framework.logger import MindPilotLogger


def make_agent():
    logger = MindPilotLogger(session_id="test_lit", verbose=False)
    llm = LLMClient(CONFIG)
    arxiv = ArXivSearchTool(max_results=3, logger=logger)
    memory = MemoryStore(logger=logger)
    return LiteratureAgent(CONFIG, llm, arxiv, memory, logger)


class TestArXivTool(unittest.TestCase):
    def test_search_returns_list(self):
        arxiv = ArXivSearchTool(max_results=3)
        papers = arxiv.search("attention mechanism")
        self.assertIsInstance(papers, list)
        self.assertGreater(len(papers), 0)

    def test_paper_fields(self):
        arxiv = ArXivSearchTool(max_results=2)
        papers = arxiv.search("transformer")
        for p in papers:
            self.assertIsInstance(p, Paper)
            self.assertTrue(p.title)
            self.assertTrue(p.arxiv_id)
            self.assertIsInstance(p.authors, list)

    def test_relevance_score_range(self):
        arxiv = ArXivSearchTool(max_results=3)
        papers = arxiv.search("BERT language model")
        for p in papers:
            self.assertGreaterEqual(p.relevance_score, 0.0)
            self.assertLessEqual(p.relevance_score, 1.0)

    def test_mock_papers_fallback(self):
        arxiv = ArXivSearchTool(max_results=3)
        mocks = arxiv._mock_papers("test query", 3)
        self.assertEqual(len(mocks), 3)


class TestKnowledgeGraph(unittest.TestCase):
    def test_add_paper_creates_nodes(self):
        kg = LightKnowledgeGraph()
        p = Paper("2024.00001", "Test Paper", ["Alice", "Bob"], "Abstract text",
                  "2024-01-01", ["cs.AI"], "https://arxiv.org", "https://arxiv.org/pdf",
                  relevance_score=0.8)
        kg.add_paper(p)
        self.assertGreater(len(kg.nodes), 0)
        self.assertGreater(len(kg.edges), 0)

    def test_multi_hop_query(self):
        kg = LightKnowledgeGraph()
        p = Paper("2024.00002", "Attention Paper", ["Carol"], "Abstract",
                  "2024-01-01", ["cs.LG"], "https://arxiv.org", "https://arxiv.org/pdf",
                  relevance_score=0.7)
        kg.add_paper(p)
        results = kg.multi_hop_query("Attention", hops=2)
        self.assertIsInstance(results, list)

    def test_stats(self):
        kg = LightKnowledgeGraph()
        stats = kg.stats()
        self.assertIn("nodes", stats)
        self.assertIn("edges", stats)


class TestLiteratureAgent(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_run_returns_dict(self):
        result = self.agent.run("attention mechanism", "attention")
        self.assertIsInstance(result, dict)
        self.assertIn("papers", result)
        self.assertIn("literature_review", result)

    def test_metrics_present(self):
        result = self.agent.run("transformer model")
        self.assertIn("metrics", result)
        self.assertIn("recall@5", result["metrics"])

    def test_knowledge_graph_stats(self):
        result = self.agent.run("graph neural network")
        self.assertIn("knowledge_graph", result)
        self.assertIn("nodes", result["knowledge_graph"])

    def test_recall_at_k_range(self):
        result = self.agent.run("deep learning")
        r5 = result["metrics"]["recall@5"]
        self.assertGreaterEqual(r5, 0.0)
        self.assertLessEqual(r5, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
