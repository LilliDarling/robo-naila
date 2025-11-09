"""Text normalization patterns for TTS preprocessing

This module contains regex patterns and mappings for text normalization.
Keeping patterns separate improves maintainability and allows for easy updates.
"""

import re

# Abbreviation expansion patterns
# Format: (pattern, replacement, flags)
ABBREVIATION_PATTERNS = [
    # ===== TITLES (must be followed by a name/word) =====
    (r'\bDr\.(?=\s+[A-Z])', 'Doctor', 0),
    (r'\bMr\.(?=\s+[A-Z])', 'Mister', 0),
    (r'\bMrs\.(?=\s+[A-Z])', 'Missus', 0),
    (r'\bMs\.(?=\s+[A-Z])', 'Miss', 0),
    (r'\bProf\.(?=\s+[A-Z])', 'Professor', 0),

    # ===== NAME SUFFIXES (preceded by a name) =====
    (r'(?<=\s)Sr\.(?=\s|$|,)', 'Senior', 0),
    (r'(?<=\s)Jr\.(?=\s|$|,)', 'Junior', 0),
    (r'(?<=\s)Esq\.(?=\s|$|,)', 'Esquire', 0),
    (r'(?<=\s)PhD\.(?=\s|$|,)', 'P H D', 0),
    (r'(?<=\s)MD\.(?=\s|$|,)', 'M D', 0),

    # ===== STREET TYPES =====
    (r'\bSt\.(?=\s|$)', 'Street', 0),
    (r'\bAve\.(?=\s|$)', 'Avenue', 0),
    (r'\bBlvd\.(?=\s|$)', 'Boulevard', 0),
    (r'\bRd\.(?=\s|$)', 'Road', 0),
    (r'\bDr\.(?=\s|$)', 'Drive', 0),  # Drive (not Doctor without capital letter)
    (r'\bLn\.(?=\s|$)', 'Lane', 0),
    (r'\bPl\.(?=\s|$)', 'Place', 0),
    (r'\bCt\.(?=\s|$)', 'Court', 0),
    (r'\bPkwy\.(?=\s|$)', 'Parkway', 0),

    # ===== DIRECTIONS =====
    (r'\bN\.(?=\s)', 'North', 0),
    (r'\bS\.(?=\s)', 'South', 0),
    (r'\bE\.(?=\s)', 'East', 0),
    (r'\bW\.(?=\s)', 'West', 0),
    (r'\bNE\.(?=\s)', 'Northeast', 0),
    (r'\bNW\.(?=\s)', 'Northwest', 0),
    (r'\bSE\.(?=\s)', 'Southeast', 0),
    (r'\bSW\.(?=\s)', 'Southwest', 0),

    # ===== COMMON LATIN ABBREVIATIONS =====
    (r'\betc\.(?=\s|$)', 'et cetera', re.IGNORECASE),
    (r'\be\.g\.(?=\s)', 'for example', re.IGNORECASE),
    (r'\bi\.e\.(?=\s)', 'that is', re.IGNORECASE),
    (r'\bvs\.(?=\s)', 'versus', re.IGNORECASE),
    (r'\bcf\.(?=\s)', 'compare', re.IGNORECASE),
    (r'\bet al\.(?=\s|$)', 'and others', re.IGNORECASE),

    # ===== TIME PERIODS =====
    (r'\ba\.m\.(?=\s|$)', 'A M', re.IGNORECASE),
    (r'\bp\.m\.(?=\s|$)', 'P M', re.IGNORECASE),
    (r'\bAM(?=\s|$)', 'A M', 0),
    (r'\bPM(?=\s|$)', 'P M', 0),

    # ===== UNITS OF MEASUREMENT =====
    (r'\bft\.(?=\s|$)', 'feet', 0),
    (r'\bin\.(?=\s|$)', 'inches', 0),
    (r'\byd\.(?=\s|$)', 'yards', 0),
    (r'\blb\.(?=\s|$)', 'pound', 0),
    (r'\blbs\.(?=\s|$)', 'pounds', 0),
    (r'\boz\.(?=\s|$)', 'ounce', 0),
    (r'\bmi\.(?=\s|$)', 'miles', 0),
    (r'\bkm\.(?=\s|$)', 'kilometers', 0),
    (r'\bcm\.(?=\s|$)', 'centimeters', 0),
    (r'\bmm\.(?=\s|$)', 'millimeters', 0),
    (r'\bkg\.(?=\s|$)', 'kilograms', 0),
    (r'\bg\.(?=\s|$)', 'grams', 0),

    # ===== BUSINESS/ORGANIZATION =====
    (r'\bInc\.(?=\s|$)', 'Incorporated', 0),
    (r'\bCorp\.(?=\s|$)', 'Corporation', 0),
    (r'\bLtd\.(?=\s|$)', 'Limited', 0),
    (r'\bLLC\.(?=\s|$)', 'L L C', 0),
    (r'\bCo\.(?=\s)', 'Company', 0),

    # ===== DAYS OF WEEK (abbreviated) =====
    (r'\bMon\.(?=\s|$)', 'Monday', 0),
    (r'\bTue\.(?=\s|$)', 'Tuesday', 0),
    (r'\bTues\.(?=\s|$)', 'Tuesday', 0),
    (r'\bWed\.(?=\s|$)', 'Wednesday', 0),
    (r'\bThu\.(?=\s|$)', 'Thursday', 0),
    (r'\bThur\.(?=\s|$)', 'Thursday', 0),
    (r'\bThurs\.(?=\s|$)', 'Thursday', 0),
    (r'\bFri\.(?=\s|$)', 'Friday', 0),
    (r'\bSat\.(?=\s|$)', 'Saturday', 0),
    (r'\bSun\.(?=\s|$)', 'Sunday', 0),

    # ===== MONTHS (abbreviated) =====
    (r'\bJan\.(?=\s)', 'January', 0),
    (r'\bFeb\.(?=\s)', 'February', 0),
    (r'\bMar\.(?=\s)', 'March', 0),
    (r'\bApr\.(?=\s)', 'April', 0),
    (r'\bJun\.(?=\s)', 'June', 0),
    (r'\bJul\.(?=\s)', 'July', 0),
    (r'\bAug\.(?=\s)', 'August', 0),
    (r'\bSep\.(?=\s)', 'September', 0),
    (r'\bSept\.(?=\s)', 'September', 0),
    (r'\bOct\.(?=\s)', 'October', 0),
    (r'\bNov\.(?=\s)', 'November', 0),
    (r'\bDec\.(?=\s)', 'December', 0),
]

# Month names for date normalization
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Currency symbols to word mappings
CURRENCY_SYMBOLS = {
    '€': 'euros',
    '£': 'pounds',
    '¥': 'yen',
    '₹': 'rupees',
    '₽': 'rubles',
}

# Special character replacements for speech
SPECIAL_CHAR_REPLACEMENTS = {
    '&': ' and ',
    '+': ' plus ',
    '=': ' equals ',
    '%': ' percent',
    '#': ' number ',
    '@': ' at ',
}

# URL pattern
URL_PATTERN = r'https?://[^\s]+'

# Email pattern
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Date pattern: MM/DD/YYYY or M/D/YYYY
DATE_PATTERN = r'(\d{1,2})/(\d{1,2})/(\d{4})'

# Time pattern: HH:MM AM/PM or HH:MM
TIME_PATTERN = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?'

# Dollar amount pattern: $123.45
DOLLAR_PATTERN = r'\$(\d+(?:\.\d{1,2})?)'

# Number pattern (integers and decimals)
NUMBER_PATTERN = r'\b\d+(?:\.\d+)?\b'
