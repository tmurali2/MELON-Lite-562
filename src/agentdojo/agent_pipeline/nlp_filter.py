#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Optional
import json
import re

# referred to https://thepythoncode.com/article/credit-card-validation-in-python
# referred to https://www.geeksforgeeks.org/python/parsing-and-processing-url-using-python-regex/
# referred to https://www.geeksforgeeks.org/python/check-if-email-address-valid-or-not-in-python
# referred to https://docs.python.org/3/library/re.html
# referred to https://stackoverflow.com/questions/36533356/how-does-this-regular-expression-d-13-16-b-work


# Validate credit card number using Luhn Algorithm
def luhn_check(number: str) -> bool:
    def digits_of(n):
        return [int(d) for d in str(n)]

    # Strip non-digit characters and validate length
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    # Luhn algorithm
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))
    return checksum % 10 == 0


# validating emails, URLs, and credit cards
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", re.I)
URL_REGEX = re.compile(r"(\w+)://([\w\-.]+)/(\w+)\.(\w+)", re.I)
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# risky URLs
RISKY_URL_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "ow.ly", "goo.gl", "is.gd", "buff.ly", "rebrand.ly", "lnkd.in", "s.id", "t.ly"}
RISKY_PASTEBINS  = {"pastebin.com", "hastebin.com", "ghostbin.com", "rentry.co", "paste.ee"}

NEGATION_CUES = {"do not", "don't", "never", "refuse", "must not", "should not", "avoid", "prohibit"}


@dataclass
class RuleConfig:
    credential_terms: List[str] = field(default_factory=lambda: [
        "password", "passcode", "otp", "one-time code", "verification code", "2fa", "security answer"
    ])
    financial_terms: List[str] = field(default_factory=lambda: [
        "send to", "transfer", "deposit", "wire", "payout", "wallet", "crypto address"
    ])
    exfil_terms: List[str] = field(default_factory=lambda: [
        "email me", "share externally", "share", "upload", "post publicly"
    ])
    hijack_terms: List[str] = field(default_factory=lambda: [
        "ignore previous instructions", "override", "follow these steps instead"
    ])

@dataclass
class FilterResult:
    is_risky: bool
    weight: float
    matched_terms: List[str]


class NLPFilter:
    def __init__(self, config: Optional[RuleConfig] = None):
        self.conf = config or RuleConfig()

    def analyze(self, text: str) -> FilterResult:
        lower_case = text.lower()
        matched_terms: List[str] = []
        weight: float = 0.0

        # combine all terms
        all_terms = (
            self.conf.credential_terms +
            self.conf.financial_terms +
            self.conf.exfil_terms +
            self.conf.hijack_terms
        )

        for term in all_terms:
            if term in lower_case:
                # check if term appears in negated context
                idx = lower_case.find(term)
                # 20 is chosen to capture most negation cues
                window = lower_case[max(0, idx - 20):idx]
                if any(cue in window for cue in NEGATION_CUES):
                    continue
                matched_terms.append(term)
                weight += 1.0

        if EMAIL_REGEX.search(text):
            matched_terms.append("email_detected")
            weight += 0.5

        # example url_regex output: [('http', 'www.example.com', 'index', 'html')]
        for match in URL_REGEX.findall(text):
            domain = match[1].lower()
            if domain.startswith("www."):
                domain = domain[4:]  # strip "www." prefix
            if domain in RISKY_URL_SHORTENERS:
                matched_terms.append(f"risky_shortener:{domain}")
                weight += 1.0
            elif domain in RISKY_PASTEBINS:
                matched_terms.append(f"risky_pastebin:{domain}")
                weight += 1.0

        # check for credit card numbers using Luhn algorithm
        # pii detection
        for candidate in CREDIT_CARD_REGEX.findall(text):
            if luhn_check(candidate):
                matched_terms.append("credit_card_detected")
                weight += 2.0

        is_risky = len(matched_terms) > 0

        return FilterResult(
            is_risky=is_risky,
            weight=weight,
            matched_terms=matched_terms
        )

    def is_risky(self, text: str) -> bool:
        return self.analyze(text).is_risky

    def get_weight(self, text: str) -> float:
        return self.analyze(text).weight
