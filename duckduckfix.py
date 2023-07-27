from typing import Dict, List, Optional
from pydantic.class_validators import root_validator
from pydantic import BaseModel, Extra


class DuckDuckGoSearchAPIWrapper(BaseModel):

    region: Optional[str] = "wt-wt"
    safesearch: str = "moderate"
    timelimit: Optional[str] = "y"
    backend: str = "api"
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            from duckduckgo_search import DDGS  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import duckduckgo-search python package. "
                "Please install it with `pip install duckduckgo-search`."
            )
        return values

    def run(self, query: str) -> str:
        from duckduckgo_search import DDGS

        """Run query through DuckDuckGo and return results."""
        ddgs = DDGS()
        
        results = ddgs.text(
            query,
            region=self.region,
            safesearch=self.safesearch,
        )
        results = list(results)
        if len(results) == 0:
            return "No good DuckDuckGo Search Result was found"
        snippets = [result["body"] for result in results]
        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through DuckDuckGo and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = ddgs.text(
            query,
            region=self.region,
            safesearch=self.safesearch,
            time=self.timelimit
        )

        if len(results) == 0:
            return [{"Result": "No good DuckDuckGo Search Result was found"}]

        def to_metadata(result: Dict) -> Dict:
            return {
                "snippet": result["body"],
                "title": result["title"],
                "link": result["href"],
            }

        return [to_metadata(result) for result in results]