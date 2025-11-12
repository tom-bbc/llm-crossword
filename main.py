# --------------------------- Imports ---------------------------
import argparse
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.crossword.crossword import CrosswordPuzzle
from src.crossword.types import Clue, Direction
from src.crossword.utils import load_puzzle


# ---------- Main class that implements crossword helper ----------
class CrosswordHelper:
    def __init__(self) -> None:
        """Crossword helper using OpenAI model."""
        # Load OpenAI credentials from local .env file
        load_dotenv()

        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
        openai_model_name: Optional[str] = os.getenv("OPENAI_MODEL_NAME")

        # Ensure credentials are all set
        if not openai_api_key or not openai_base_url or not openai_model_name:
            raise KeyError(
                "OpenAI API key, base URL, or model name not specified "
                "in .env credentials file."
            )

        # Instantiate OpenAI model
        self.client: OpenAI = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.model_name: str = openai_model_name

        self.dev_prompt: List[ChatCompletionMessageParam] = [
            {
                "role": "developer",
                "content": (
                    "You are a helpful assistant tasked with solving a crossword "
                    "puzzle. I will provide you with a clue 'clue' and a number of "
                    "characters 'length', and you should return the word that is being "
                    "described by the clue that contains 'length' characters. "
                    "We may already know the first character of the word we want to "
                    "find. If this is the case, the caracter will be given in the "
                    "referenced in the input as 'first_char', and the returned word "
                    "should start with this character."
                ),
            }
        ]

    def run(self, puzzle: CrosswordPuzzle, max_retries: int = 3) -> CrosswordPuzzle:
        """Run the full solving process."""
        # Iterate through clues making guesses
        for clue in puzzle.clues:
            print(f"Next clue: {clue}", end="\n\n")
            guess = None

            # Use model to come up with a guess at the clue
            for _ in range(max_retries):
                guess = self.make_guess(clue, puzzle)

                if guess is not None:
                    break

            if guess is None:
                print(f"Error: could not come up with guess for clue: {clue}")
                exit(1)

            # Set guess within puzzle
            puzzle.set_clue_chars(clue, guess)
            print(f"Updated puzzle: \n{puzzle}", end="\n\n")

        # Return filled puzzle
        return puzzle

    def make_guess(self, clue: Clue, puzzle: CrosswordPuzzle) -> Optional[List[str]]:
        """Make a single guess to the answer to a clue using the OpenAI model."""
        # Find any overlapping clues that have been filled already
        overlapping_first_char = self.get_single_overlap(clue, puzzle)

        # Send clue to model to generate a potential solution
        if overlapping_first_char is not None:
            clue_prompt = (
                f"'clue': '{clue.text}', "
                f"\n'length': '{clue.length}', "
                f"\n'first_char': '{overlapping_first_char}'"
            )
        else:
            clue_prompt = f"'clue': '{clue.text}', \n'length': '{clue.length}'"

        helper_response = self.get_response(clue_prompt)

        # Validate and format model response
        if helper_response is None:
            print("Warning: no response returned by OpenAI model. Retrying.")
            return None

        guess = self.format_response(helper_response)

        if len(guess) != clue.length:
            print(
                f"Warning: incorrect length guess returned by model. Retrying. "
                f"Guess: {guess}"
            )
            return None

        return guess

    @staticmethod
    def get_single_overlap(clue: Clue, puzzle: CrosswordPuzzle) -> Optional[str]:
        """
        Identify when the first character of our clue overlaps with
        an existing solved word.
        """
        # Extract the character we want to find any overlaps for
        clue_start_row = clue.row
        clue_start_col = clue.col
        known_first_char = None

        # Iterate through other clues in the puzzle to find overlaps
        for clue in puzzle.clues:
            if (
                (clue_start_row, clue_start_col) in clue.cells()
                and clue.answered is True
                and clue.answer is not None
            ):
                overlapping_char_index = clue.cells().index(
                    (clue_start_row, clue_start_col)
                )

                clue_chars = puzzle.get_current_clue_chars(clue)
                known_first_char = clue_chars[overlapping_char_index]

        if known_first_char is not None:
            print(f"First character already known: {known_first_char}", end="\n\n")

        return known_first_char

    @staticmethod
    def get_all_overlapping(clue: Clue, puzzle: CrosswordPuzzle) -> List[str]:
        """
        Identify when any character of our clue overlaps with
        an existing solved word.
        """

        # Get any and all overlapping characters with solved words and their index
        clue_start_row = clue.row
        clue_start_col = clue.col
        clue_direction = clue.direction

        if clue_direction == Direction.ACROSS:
            clue_rows = list(range(clue_start_row, clue.length))
            clue_cols = [clue_start_col]
        else:
            clue_rows = [clue_start_row]
            clue_cols = list(range(clue_start_col, clue.length))

        # TODO: Iterate through coordinates from clue_rows and clue_columns and use the
        # the 'get_single_overlap' method to get any overlapping character values

        return []

    def get_response(self, prompt: str) -> Optional[str]:
        """Send prompt to OpenAI model to generate a response."""

        # Format prompt as chat message in history
        chat_message = self.dev_prompt + [{"role": "user", "content": prompt}]
        print(f"Prompting model with chat: {chat_message}")

        # Send prompt to OpenAI model and get response
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_message,
            )

            generated_response = completion.choices[0].message.content

        except Exception as err:
            print(f"Error in OpenAI call: {err}")
            generated_response = None

        return generated_response

    @staticmethod
    def format_response(response: str) -> list:
        """Remove reasoning steps from an OpenAI response to only return the answer."""
        resp_no_reasoning = response.split("</reasoning>")[-1]
        resp_no_reasoning = resp_no_reasoning.lower()

        guessed_characters = list(resp_no_reasoning)
        guessed_characters = [char for char in guessed_characters if char.isalnum()]

        return guessed_characters


# --------------------------- Entrypoint ---------------------------
if __name__ == "__main__":
    # Run with: 'python main.py -p data/medium.json'
    # Get path of puzzle from command-line
    parser = argparse.ArgumentParser(
        prog="CrosswordHelper",
        description="OpenAI-powered solver for crossword puzzles",
    )
    parser.add_argument(
        "-p", "--puzzle_path", required=False, default="data/easy.json", type=str
    )

    args = parser.parse_args()
    puzzle_path = args.puzzle_path

    # Load in crossword puzzle
    puzzle = load_puzzle(puzzle_path)

    # Instantiate crossword helper
    crossword_helper = CrosswordHelper()

    # Pass puzzle to helper
    final_puzzle = crossword_helper.run(puzzle)
    print(f"Full puzzle: \n{final_puzzle}", end="\n\n")

    # Validate filled puzzle
    completed = final_puzzle.validate_all()
    print(f"Success: {completed}")
