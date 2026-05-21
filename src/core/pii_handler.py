import logging
# Notice the two new imports added here: Pattern and PatternRecognizer
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger("loan_underwriter")

class PIIHandler:
    def __init__(self):
        try:
            # Initialize Presidio Analyzer
            self.analyzer = AnalyzerEngine()
            
            # --- NEW CUSTOM LOGIC FOR INDIAN PAN ---
            # 1. Define the Regex pattern for a PAN card (5 letters, 4 numbers, 1 letter)
            pan_pattern = Pattern(name="pan_pattern", regex=r"[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.85)
            
            # 2. Create a recognizer that tells Presidio to label this pattern as "IN_PAN"
            pan_recognizer = PatternRecognizer(supported_entity="IN_PAN", patterns=[pan_pattern])
            
            # 3. Add this new recognizer to the analyzer engine you just created
            self.analyzer.registry.add_recognizer(pan_recognizer)
            # ---------------------------------------

            # Initialize Anonymizer
            self.anonymizer = AnonymizerEngine()
            
            logger.info("Presidio PII engines initialized successfully with custom PAN support.")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")
            self.analyzer = None
            self.anonymizer = None

    def redact(self, text: str) -> str:
        """Detects and redacts PII from a given string."""
        if not text or not self.analyzer or not self.anonymizer:
            return text
            
        try:
            # Analyze text for PII entities (PERSON, PHONE, EMAIL, US_SSN, IN_PAN, etc.)
            results = self.analyzer.analyze(text=text, language='en')
            
            # Anonymize the detected entities (e.g., replaces name with <PERSON>, PAN with <IN_PAN>)
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            logger.error(f"Error during PII redaction: {e}")
            return text

# Create a singleton instance to be used across the app
pii_handler = PIIHandler()