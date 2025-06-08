import requests
import argparse
import json
from typing import Dict, Any, List
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import tldextract
import time
from urllib.parse import urlparse
import re

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_text(self, url: str) -> str:
        """Extract main content text from a webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text from paragraphs and headings
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            text = ' '.join([elem.get_text().strip() for elem in text_elements])
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            return text[:5000]  # Limit text length
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

class ResearchAssistant:
    def __init__(self, model: str = "deepseek-r1:8b"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
        self.scraper = WebScraper()
        self.ddgs = DDGS()
        
    def _search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo."""
        try:
            results = []
            for r in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r['title'],
                    'url': r['link'],
                    'snippet': r['body']
                })
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _create_research_prompt(self, question: str, web_results: List[Dict[str, str]]) -> str:
        """Create a structured prompt for deep research with web results."""
        web_context = "\n\nWeb Search Results:\n"
        for i, result in enumerate(web_results, 1):
            web_context += f"\n{i}. {result['title']}\n"
            web_context += f"   URL: {result['url']}\n"
            web_context += f"   Summary: {result['snippet']}\n"
        
        return f"""You are a research assistant tasked with providing a comprehensive analysis of the following question:

{question}

{web_context}

Please provide a detailed response that includes:
1. Key concepts and definitions
2. Historical context (if relevant)
3. Current state of knowledge
4. Different perspectives or viewpoints
5. Supporting evidence and examples
6. Potential implications
7. Areas for further research
8. Citations and sources (include URLs from the web results)

Structure your response in a clear, academic format with appropriate sections and subsections."""

    def _create_analysis_prompt(self, question: str, initial_response: str, web_results: List[Dict[str, str]]) -> str:
        """Create a prompt for deeper analysis of the initial response."""
        web_context = "\n\nWeb Search Results:\n"
        for i, result in enumerate(web_results, 1):
            web_context += f"\n{i}. {result['title']}\n"
            web_context += f"   URL: {result['url']}\n"
            web_context += f"   Summary: {result['snippet']}\n"
        
        return f"""Based on the following research question, initial response, and web search results, provide a deeper analysis:

Question: {question}

Initial Response:
{initial_response}

{web_context}

Please provide:
1. Critical analysis of the information presented
2. Identification of any gaps or limitations
3. Connections to related fields or concepts
4. Practical applications or implications
5. Recommendations for further investigation
6. Fact-checking against web sources
7. Additional insights from web sources"""

    def research(self, question: str, depth: int = 2, max_web_results: int = 5) -> Dict[str, Any]:
        """
        Conduct deep research on a given question with web search capabilities.
        
        Args:
            question: The research question to investigate
            depth: Number of analysis iterations (default: 2)
            max_web_results: Maximum number of web results to include (default: 5)
            
        Returns:
            Dictionary containing the research results and analysis
        """
        results = {
            "question": question,
            "web_results": [],
            "initial_research": None,
            "analysis": [],
            "final_conclusions": None
        }
        
        # Perform web search
        print("Searching the web...")
        web_results = self._search_web(question, max_web_results)
        results["web_results"] = web_results
        
        # Scrape content from web results
        print("Scraping web content...")
        for result in web_results:
            content = self.scraper.extract_text(result['url'])
            result['content'] = content
            time.sleep(1)  # Be nice to servers
        
        # Initial research
        print("Conducting initial research...")
        initial_prompt = self._create_research_prompt(question, web_results)
        initial_response = self._query_model(initial_prompt)
        results["initial_research"] = initial_response
        
        # Deep analysis iterations
        current_response = initial_response
        for i in range(depth):
            print(f"Performing analysis iteration {i+1}...")
            analysis_prompt = self._create_analysis_prompt(question, current_response, web_results)
            analysis_response = self._query_model(analysis_prompt)
            results["analysis"].append(analysis_response)
            current_response = analysis_response
        
        # Final synthesis
        print("Generating final synthesis...")
        synthesis_prompt = f"""Based on the following research question, all previous analyses, and web search results, provide a final synthesis:

Question: {question}

Initial Research:
{results['initial_research']}

Analysis Iterations:
{json.dumps(results['analysis'], indent=2)}

Web Results:
{json.dumps(web_results, indent=2)}

Please provide:
1. A comprehensive synthesis of all findings
2. Key takeaways and conclusions
3. Practical implications
4. Recommendations for further research
5. Citations and sources
6. Fact-checking summary"""
        
        results["final_conclusions"] = self._query_model(synthesis_prompt)
        return results

    def _query_model(self, prompt: str) -> str:
        """Query the Ollama model with the given prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['response']
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Deep Research Assistant using Ollama with Web Search')
    parser.add_argument('question', help='The research question to investigate')
    parser.add_argument('--depth', type=int, default=2, help='Number of analysis iterations (default: 2)')
    parser.add_argument('--web-results', type=int, default=5, help='Maximum number of web results to include (default: 5)')
    parser.add_argument('--output', help='Output file path for saving results (optional)')
    args = parser.parse_args()
    
    assistant = ResearchAssistant()
    results = assistant.research(args.question, args.depth, args.web_results)
    
    # Print results
    print("\nResearch Results:")
    print("=" * 80)
    print(f"\nQuestion: {results['question']}")
    
    print("\nWeb Search Results:")
    print("-" * 40)
    for i, result in enumerate(results['web_results'], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Summary: {result['snippet']}")
    
    print("\nInitial Research:")
    print("-" * 40)
    print(results['initial_research'])
    
    for i, analysis in enumerate(results['analysis'], 1):
        print(f"\nAnalysis Iteration {i}:")
        print("-" * 40)
        print(analysis)
    
    print("\nFinal Conclusions:")
    print("-" * 40)
    print(results['final_conclusions'])
    
    # Save results to file if output path is provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main() 