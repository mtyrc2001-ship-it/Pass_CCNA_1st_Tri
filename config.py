# config.py
import os
from pathlib import Path

APP_TITLE = "CCNA Mastery 2026"
APP_ICON = "🚀"

# Leave empty if not set; app can show a friendly warning if needed.
# config.py
import os
from pathlib import Path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DB_FILE = Path("data/ccna_mastery.db")


DIFFICULTY_SCALE = ["Easy", "Medium", "Hard"]
DIFFICULTY_WEIGHT = {"Easy": 0.8, "Medium": 1.0, "Hard": 1.2}

DOMAIN_WEIGHTS = {
    "Network Fundamentals": 0.20,
    "Network Access": 0.20,
    "IP Connectivity": 0.25,
    "IP Services": 0.10,
    "Security Fundamentals": 0.15,
    "Automation and Programmability": 0.10,
    "Wireless": 0.10,
}

# Optional robustness: normalize weights so they sum to 1.0
_total = sum(DOMAIN_WEIGHTS.values())
if _total > 0:
    DOMAIN_WEIGHTS = {k: v / _total for k, v in DOMAIN_WEIGHTS.items()}

CCNA_BLUEPRINT = {
    "Network Fundamentals": [
        "1.1 Explain the role and function of network components",
        "1.2 Describe characteristics of network topology architectures",
        "1.3 Compare physical interface and cabling types",
        "1.4 Identify interface and cable issues",
        "1.5 Compare TCP to UDP",
        "1.6 Configure and verify IPv4 addressing",
    ],
    "Network Access": [
        "2.1 Configure and verify VLANs",
        "2.2 Configure and verify interswitch connectivity",
        "2.3 Configure and verify Layer 2 discovery protocols",
        "2.4 Configure and verify wireless LANs",
    ],
    "IP Connectivity": [
        "3.1 Interpret routing table",
        "3.2 Configure and verify IPv4 and IPv6 static routing",
        "3.3 Configure and verify OSPFv2",
    ],
    "IP Services": [
        "4.1 Configure and verify DHCP",
        "4.2 Configure and verify NTP",
        "4.3 Explain SNMP",
    ],
    "Security Fundamentals": [
        "5.1 Define key security concepts",
        "5.2 Configure and verify ACLs",
        "5.3 Configure and verify port security",
        "5.4 Explain wireless security protocols",
    ],
    "Automation and Programmability": [
        "6.1 Explain automation impacts",
        "6.2 Compare controller-based networking",
        "6.3 Describe REST APIs",
        "6.4 Interpret JSON data",
    ],
    "Wireless": [
        "WLC architecture and CAPWAP",
        "WLC show command interpretation",
        "Wireless security attack mitigation",
        "802.11 standards and channels",
    ],
}

DB_FILE = Path("data/ccna_mastery.db")
