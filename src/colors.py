# colors.py
"""
Defines ANSI escape codes for colored console output.
"""

# Reset any applied formatting
COLOR_RESET = "\033[0m"

# Standard ANSI colors
COLOR_BLACK = "\033[30m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_WHITE = "\033[37m"

# Bright/Light ANSI colors (often preferred for readability)
COLOR_BRIGHT_BLACK = "\033[90m"
COLOR_BRIGHT_RED = "\033[91m"
COLOR_BRIGHT_GREEN = "\033[92m"
COLOR_BRIGHT_YELLOW = "\033[93m"
COLOR_BRIGHT_BLUE = "\033[94m"
COLOR_BRIGHT_MAGENTA = "\033[95m"
COLOR_BRIGHT_CYAN = "\033[96m"
COLOR_BRIGHT_WHITE = "\033[97m"

# Common text styles
COLOR_BOLD = "\033[1m"
COLOR_ITALIC = "\033[3m"
COLOR_UNDERLINE = "\033[4m"
COLOR_STRIKETHROUGH = "\033[9m"

# Semantic color aliases for easier use in your application
# (Based on your previous use-case)
INFO_COLOR = COLOR_BRIGHT_GREEN      # For successful operations, [+] messages
WARNING_COLOR = COLOR_BRIGHT_YELLOW  # For warnings, [!] messages
ERROR_COLOR = COLOR_BRIGHT_RED       # For errors, [!] messages
PROMPT_COLOR = COLOR_BRIGHT_CYAN     # For user input prompts, questionary questions
HIGHLIGHT_COLOR = COLOR_BRIGHT_MAGENTA # For important numerical data (tokens, cost)
HEADER_COLOR = COLOR_BRIGHT_BLUE     # For section headers or visual separators
STREAM_COLOR = COLOR_BRIGHT_CYAN     # For streaming LLM output
