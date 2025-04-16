import logging
import uuid
from typing import List, Dict, Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import config
from db.db_manager import DBManager
from retriever.retriever import Retriever

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
            self,
            db_manager: DBManager,
            retriever: Retriever,
            max_history: int = 5,
            max_references: int = 3
    ):
        self.db = db_manager
        self.retriever = retriever
        self.max_history = max_history
        self.max_references = max_references

        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            openai_api_key=config.openai.api_key
        )

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news assistant. Use the provided context to answer questions.
            Base your answers on the given news references and maintain a natural conversation flow.
            If you're unsure or don't have relevant information, say so.
            
            Current conversation history and context:
            {chat_history}
            
            Relevant news references:
            {references}
            """),
            ("human", "{query}")
        ])

    async def create_session(self) -> Dict[str, Any]:
        session_data = self.db.create_chat_session()
        return {
            "id": session_data["id"],
            "created_at": session_data["created_at"]
        }

    async def process_message(self, session_id: uuid.UUID, query: str) -> Dict[str, Any]:
        try:
            # Update session activity
            self.db.update_chat_session_activity(session_id)

            # Store user message
            user_message = self.db.create_chat_message(
                session_id=session_id,
                role="user",
                content=query
            )

            # Get chat history
            history = self._format_chat_history(
                self.db.get_chat_history(session_id, self.max_history * 2)
            )

            # Process query and generate response
            relevant_chunks = await self._get_relevant_chunks(query, history)
            response = await self._generate_response(query, history, relevant_chunks)

            # Store assistant message
            assistant_message = self.db.create_chat_message(
                session_id=session_id,
                role="assistant",
                content=response["content"]
            )

            # Store references
            references = [
                {
                    "session_id": session_id,
                    "message_id": assistant_message["id"],  # Updated to use dictionary
                    "news_id": ref["id"],
                    "relevance_score": ref["score"],
                    "content_snippet": ref["content"][:500]
                }
                for ref in response["references"]
            ]
            self.db.create_chat_references(references)

            return {
                "session_id": str(session_id),
                "response": response["content"],
                "references": response["references"]
            }

        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise

    async def get_session_info(self, session_id: uuid.UUID) -> Dict[str, Any]:
        return self.db.get_session_info(session_id)

    def _format_chat_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in reversed(messages)
        ]

    async def _get_relevant_chunks(
            self,
            query: str,
            history: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant news chunks considering chat history"""
        # Combine current query with recent context
        context_query = query
        if history:
            recent_context = " ".join(
                msg["content"] for msg in history[-2:]
                if msg["role"] == "user"
            )
            context_query = f"{recent_context} {query}"

        # Get relevant chunks
        results = await self.retriever.search(
            query=context_query,
            top_k=self.max_references
        )

        return results

    async def _generate_response(
            self,
            query: str,
            history: List[Dict[str, str]],
            references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate response using LLM"""
        # Format history and references for prompt
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in history[-self.max_history:]
        )

        references_text = "\n\n".join(
            f"Source {i + 1}:\n{ref['content'][:500]}..."
            for i, ref in enumerate(references)
        )

        # Generate response
        messages = self.chat_prompt.format_messages(
            chat_history=history_text,
            references=references_text,
            query=query
        )

        response = await self.llm.apredict_messages(messages)

        return {
            "content": response.content,
            "references": references
        }
