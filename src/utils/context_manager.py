# src/utils/token_manager.py
import copy
import logging
from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from src.config import load_yaml_config

logger = logging.getLogger(__name__)


def get_search_config():
    config = load_yaml_config("conf.yaml")
    search_config = config.get("MODEL_TOKEN_LIMITS", {})
    return search_config


class ContextManager:
    """Context manager and compression class"""

    def __init__(self, token_limit: int, preserve_prefix_message_count: int = 0):
        """
        Initialize ContextManager

        Args:
            token_limit: Maximum token limit
            preserve_prefix_message_count: Number of messages to preserve at the beginning of the context
        """
        self.token_limit = token_limit
        self.preserve_prefix_message_count = preserve_prefix_message_count

    def count_tokens(self, messages: List[BaseMessage]) -> int:
        """
        Count tokens in message list

        Args:
            messages: List of messages

        Returns:
            Number of tokens
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self._count_message_tokens(message)
        return total_tokens

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """
        Count tokens in a single message

        Args:
            message: Message object

        Returns:
            Number of tokens
        """
        token_count = 0

        # Count tokens in content field
        content = getattr(message, "content", None)
        if content:
            # Optimize type check by assuming most content will be str (breaks fast, avoids isinstance overhead)
            if type(content) is str:
                token_count += self._count_text_tokens(content)

        # Count role-related tokens using direct attribute access
        msg_type = getattr(message, "type", None)
        if msg_type:
            token_count += self._count_text_tokens(msg_type)

        # Special handling for different message types (order checks using type() for single dispatch speed)
        mtype = type(message)
        if mtype is SystemMessage:
            token_count = int(token_count * 1.1)
        elif mtype is AIMessage:
            token_count = int(token_count * 1.2)
        elif mtype is ToolMessage:
            token_count = int(token_count * 1.3)
        # HumanMessage matches default estimation

        # Process additional information in additional_kwargs
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if additional_kwargs:
            # Faster conversion to str for dicts
            extra_str = str(additional_kwargs)
            token_count += self._count_text_tokens(extra_str)
            # If there are tool_calls, add estimation
            if (
                isinstance(additional_kwargs, dict)
                and "tool_calls" in additional_kwargs
            ):
                token_count += 50  # Add estimation for function call information

        # Ensure at least 1 token
        return max(1, token_count)

    def _count_text_tokens(self, text: str) -> int:
        """
        Count tokens in text with different calculations for English and non-English characters.
        English characters: 4 characters ≈ 1 token
        Non-English characters (e.g., Chinese): 1 character ≈ 1 token

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        english_chars = 0
        non_english_chars = 0

        for char in text:
            # Check if character is ASCII (English letters, digits, punctuation)
            if ord(char) < 128:
                english_chars += 1
            else:
                non_english_chars += 1

        # Calculate tokens: English at 4 chars/token, others at 1 char/token
        english_tokens = english_chars // 4
        non_english_tokens = non_english_chars

        return english_tokens + non_english_tokens

    def is_over_limit(self, messages: List[BaseMessage]) -> bool:
        """
        Check if messages exceed token limit

        Args:
            messages: List of messages

        Returns:
            Whether limit is exceeded
        """
        return self.count_tokens(messages) > self.token_limit

    def compress_messages(self, state: dict) -> List[BaseMessage]:
        """
        Compress messages to fit within token limit

        Args:
            state: state with original messages

        Returns:
            Compressed state with compressed messages
        """
        # If not set token_limit, return original state
        if self.token_limit is None:
            logger.info("No token_limit set, the context management doesn't work.")
            return state

        if not isinstance(state, dict) or "messages" not in state:
            logger.warning("No messages found in state")
            return state

        messages = state["messages"]

        if not self.is_over_limit(messages):
            return state

        # 2. Compress messages
        compressed_messages = self._compress_messages(messages)

        logger.info(
            f"Message compression completed: {self.count_tokens(messages)} -> {self.count_tokens(compressed_messages)} tokens"
        )

        state["messages"] = compressed_messages
        return state

    def _compress_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Compress compressible messages

        Args:
            messages: List of messages to compress

        Returns:
            Compressed message list
        """

        available_token = self.token_limit
        prefix_messages = []
        num_prefix = min(self.preserve_prefix_message_count, len(messages))
        # 1. Preserve head messages of specified length to retain system prompts and user input
        for i in range(num_prefix):
            cur_token_cnt = self._count_message_tokens(messages[i])
            if available_token > 0 and available_token >= cur_token_cnt:
                prefix_messages.append(messages[i])
                available_token -= cur_token_cnt
            elif available_token > 0:
                # Truncate content to fit available tokens
                prefix_messages.append(
                    self._truncate_message_content(messages[i], available_token)
                )
                return prefix_messages
            else:
                break

        # 2. Compress subsequent messages from the tail, some messages may be discarded
        # Instead of slicing and creating a new list, just iterate directly over the tail
        suffix_messages = []
        n = len(messages)
        for idx in range(n - 1, num_prefix - 1, -1):
            cur_token_cnt = self._count_message_tokens(messages[idx])
            if cur_token_cnt > 0 and available_token >= cur_token_cnt:
                # Prepend: build as reversed list; insert at front (faster than list concatenation in loop)
                suffix_messages.insert(0, messages[idx])
                available_token -= cur_token_cnt
            elif available_token > 0:
                suffix_messages.insert(
                    0, self._truncate_message_content(messages[idx], available_token)
                )
                return prefix_messages + suffix_messages
            else:
                break

        return prefix_messages + suffix_messages

    def _truncate_message_content(
        self, message: BaseMessage, max_tokens: int
    ) -> BaseMessage:
        """
        Truncate message content while preserving all other attributes by copying the original message
        and only modifying its content attribute.

        Args:
            message: The message to truncate
            max_tokens: Maximum number of tokens to keep

        Returns:
            New message instance with truncated content
        """
        # Create a deep copy of the original message to preserve all attributes
        truncated_message = copy.deepcopy(message)
        # Truncate only the content attribute (skip assignment if possible)
        if hasattr(truncated_message, "content") and truncated_message.content:
            truncated_message.content = truncated_message.content[:max_tokens]
        return truncated_message

    def _create_summary_message(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Create summary for messages

        Args:
            messages: Messages to summarize

        Returns:
            Summary message
        """
        # TODO: summary implementation
        pass
