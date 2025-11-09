"""Text normalization utilities for TTS preprocessing"""

import logging
import re
from typing import Optional

from num2words import num2words


logger = logging.getLogger(__name__)


class TextNormalizer:
    """Normalize text for natural TTS synthesis

    Handles:
    - Numbers to words conversion
    - Date and time formatting
    - Currency formatting
    - Abbreviations
    - Special characters
    - URL/email handling
    """

    def __init__(self, language: str = "en"):
        """Initialize the text normalizer

        Args:
            language: Language code for num2words (default: en)
        """
        self.language = language

        # Common abbreviations mapping
        self.abbreviations = {
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Ms.": "Miss",
            "Prof.": "Professor",
            "Sr.": "Senior",
            "Jr.": "Junior",
            "St.": "Saint",
            "Ave.": "Avenue",
            "Blvd.": "Boulevard",
            "Rd.": "Road",
            "etc.": "et cetera",
            "vs.": "versus",
            "e.g.": "for example",
            "i.e.": "that is",
        }

    def normalize(self, text: str) -> str:
        """Apply all normalization steps to text

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text suitable for TTS
        """
        if not text:
            return ""

        # Apply normalizations in order
        text = self._normalize_abbreviations(text)
        text = self._normalize_urls_emails(text)
        text = self._normalize_currency(text)
        text = self._normalize_numbers(text)
        text = self._normalize_dates(text)
        text = self._normalize_time(text)
        text = self._normalize_whitespace(text)
        text = self._clean_special_chars(text)

        return text.strip()

    def _normalize_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text)
        return text

    def _normalize_urls_emails(self, text: str) -> str:
        """Remove or simplify URLs and emails"""
        # Remove URLs
        text = re.sub(
            r'https?://[^\s]+',
            'link',
            text
        )

        # Simplify email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'email address',
            text
        )

        return text

    def _normalize_currency(self, text: str) -> str:
        """Convert currency symbols to words"""
        # Dollar amounts: $123.45 -> one hundred twenty-three dollars and forty-five cents
        def replace_dollars(match):
            amount_str = match.group(1)
            try:
                amount = float(amount_str)
                dollars = int(amount)
                cents = int(round((amount - dollars) * 100))

                if cents == 0:
                    if dollars == 1:
                        return "one dollar"
                    return f"{num2words(dollars, lang=self.language)} dollars"
                else:
                    dollar_words = num2words(dollars, lang=self.language) if dollars != 0 else "zero"
                    cent_words = num2words(cents, lang=self.language)
                    return f"{dollar_words} dollars and {cent_words} cents"
            except (ValueError, OverflowError):
                return match.group(0)

        text = re.sub(r'\$(\d+(?:\.\d{1,2})?)', replace_dollars, text)

        # Simple currency symbols
        text = text.replace('€', 'euros')
        text = text.replace('£', 'pounds')
        text = text.replace('¥', 'yen')

        return text

    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to words"""
        def replace_number(match):
            number_str = match.group(0)
            try:
                # Handle decimal numbers
                if '.' in number_str:
                    number = float(number_str)
                    # For decimal numbers, convert to cardinal form
                    return num2words(number, lang=self.language)
                else:
                    number = int(number_str)
                    # For integers, use cardinal form
                    return num2words(number, lang=self.language)
            except (ValueError, OverflowError):
                # If conversion fails, return original
                return number_str

        # Match numbers (including decimals) that aren't part of dates/times
        # This is a simplified pattern - more complex logic could be added
        text = re.sub(r'\b\d+(?:\.\d+)?\b', replace_number, text)

        return text

    def _normalize_dates(self, text: str) -> str:
        """Convert date formats to spoken form"""
        # Pattern: MM/DD/YYYY or M/D/YYYY
        def replace_date(match):
            month, day, year = match.groups()
            try:
                month_num = int(month)
                day_num = int(day)
                year_num = int(year)

                # Month names
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]

                if 1 <= month_num <= 12:
                    month_name = month_names[month_num - 1]
                    # Convert day to ordinal
                    day_ordinal = num2words(day_num, lang=self.language, to='ordinal')
                    # Handle year
                    if year_num >= 2000:
                        year_words = num2words(year_num, lang=self.language)
                    else:
                        # Split year into two parts for better pronunciation (e.g., 1999 -> nineteen ninety-nine)
                        first_part = year_num // 100
                        second_part = year_num % 100
                        if second_part == 0:
                            year_words = num2words(first_part, lang=self.language) + " hundred"
                        else:
                            year_words = f"{num2words(first_part, lang=self.language)} {num2words(second_part, lang=self.language)}"

                    return f"{month_name} {day_ordinal}, {year_words}"
            except (ValueError, IndexError):
                pass
            return match.group(0)

        text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', replace_date, text)

        return text

    def _normalize_time(self, text: str) -> str:
        """Convert time formats to spoken form"""
        # Pattern: HH:MM AM/PM or HH:MM
        def replace_time(match):
            hour, minute = match.groups()[:2]
            am_pm = match.group(3) if len(match.groups()) > 2 else None

            try:
                hour_num = int(hour)
                minute_num = int(minute)

                # Convert to words
                if minute_num == 0:
                    time_words = f"{num2words(hour_num, lang=self.language)} o'clock"
                else:
                    hour_words = num2words(hour_num, lang=self.language)
                    minute_words = num2words(minute_num, lang=self.language)
                    time_words = f"{hour_words} {minute_words}"

                if am_pm:
                    time_words += f" {am_pm}"

                return time_words
            except ValueError:
                return match.group(0)

        # Match time with optional AM/PM
        text = re.sub(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', replace_time, text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text

    def _clean_special_chars(self, text: str) -> str:
        """Clean or convert special characters"""
        # Keep basic punctuation that helps with prosody
        # Remove or replace problematic characters

        # Convert some symbols to words
        replacements = {
            '&': ' and ',
            '+': ' plus ',
            '=': ' equals ',
            '%': ' percent',
            '#': ' number ',
            '@': ' at ',
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Remove other problematic characters (keep letters, numbers, basic punctuation)
        # Allow: letters, numbers, spaces, . , ! ? - ' "
        text = re.sub(r'[^\w\s.,!?\-\'\"]', '', text)

        return text
