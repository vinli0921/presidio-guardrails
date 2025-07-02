"""REST API server for analyzer."""

import json
import logging
import os
from logging.config import fileConfig
from pathlib import Path
from typing import Tuple, List, Dict, Any

from flask import Flask, Response, jsonify, request
from presidio_analyzer import AnalyzerEngine, AnalyzerEngineProvider, AnalyzerRequest, RecognizerResult
from werkzeug.exceptions import HTTPException

DEFAULT_PORT = "3000"

LOGGING_CONF_FILE = "logging.ini"

WELCOME_MESSAGE = r"""
 _______  _______  _______  _______ _________ ______  _________ _______
(  ____ )(  ____ )(  ____ \(  ____ \\__   __/(  __  \ \__   __/(  ___  )
| (    )|| (    )|| (    \/| (    \/   ) (   | (  \  )   ) (   | (   ) |
| (____)|| (____)|| (__    | (_____    | |   | |   ) |   | |   | |   | |
|  _____)|     __)|  __)   (_____  )   | |   | |   | |   | |   | |   | |
| (      | (\ (   | (            ) |   | |   | |   ) |   | |   | |   | |
| )      | ) \ \__| (____/\/\____) |___) (___| (__/  )___) (___| (___) |
|/       |/   \__/(_______/\_______)\_______/(______/ \_______/(_______)
"""


class ContentAnalysisRequest:
    """Request for text content analysis compatible with FMS Guardrails Orchestrator."""
    
    def __init__(self, req_data: Dict[str, Any]):
        self.contents = req_data.get("contents", [])
        self.detector_params = req_data.get("detector_params", {})


class ContentAnalysisResponse:
    """Response for text content analysis compatible with FMS Guardrails Orchestrator."""
    
    def __init__(self, start: int, end: int, text: str, detection: str, 
                 detection_type: str, score: float, detector_id: str = None,
                 evidence: List[Dict] = None, metadata: Dict = None):
        self.start = start
        self.end = end
        self.text = text
        self.detection = detection
        self.detection_type = detection_type
        self.score = score
        self.detector_id = detector_id
        self.evidence = evidence or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "detection": self.detection,
            "detection_type": self.detection_type,
            "score": self.score,
            "evidence": self.evidence,
            "metadata": self.metadata
        }
        if self.detector_id:
            result["detector_id"] = self.detector_id
        return result


class Server:
    """HTTP Server for calling Presidio Analyzer."""

    def __init__(self):
        fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
        self.logger = logging.getLogger("presidio-analyzer")
        self.logger.setLevel(os.environ.get("LOG_LEVEL", self.logger.level))
        self.app = Flask(__name__)

        analyzer_conf_file = os.environ.get("ANALYZER_CONF_FILE")
        nlp_engine_conf_file = os.environ.get("NLP_CONF_FILE")
        recognizer_registry_conf_file = os.environ.get("RECOGNIZER_REGISTRY_CONF_FILE")

        self.logger.info("Starting analyzer engine")
        self.engine: AnalyzerEngine = AnalyzerEngineProvider(
            analyzer_engine_conf_file=analyzer_conf_file,
            nlp_engine_conf_file=nlp_engine_conf_file,
            recognizer_registry_conf_file=recognizer_registry_conf_file,
        ).create_engine()
        self.logger.info(WELCOME_MESSAGE)

        @self.app.route("/health")
        def health() -> str:
            """Return basic health probe result."""
            return "Presidio Analyzer service is up"

        @self.app.route("/api/v1/text/contents", methods=["POST"])
        def text_contents() -> Tuple[str, int]:
            """Execute the analyzer function with FMS Guardrails Orchestrator format."""
            try:
                req_data = ContentAnalysisRequest(request.get_json())
                if not req_data.contents:
                    raise Exception("No contents provided")

                # Extract parameters from detector_params
                detector_params = req_data.detector_params
                language = detector_params.get("language", "en")
                score_threshold = detector_params.get("threshold", 0.5)
                entities = detector_params.get("entities")
                
                # Process each content item
                all_results = []
                for content in req_data.contents:
                    if not content:
                        all_results.append([])
                        continue
                    
                    recognizer_result_list = self.engine.analyze(
                        text=content,
                        language=language,
                        correlation_id=detector_params.get("correlation_id"),
                        score_threshold=score_threshold,
                        entities=entities,
                        return_decision_process=detector_params.get("return_decision_process", False),
                        ad_hoc_recognizers=[],  # Not supported in this format
                        context=detector_params.get("context"),
                        allow_list=detector_params.get("allow_list"),
                        allow_list_match=detector_params.get("allow_list_match", "exact"),
                        regex_flags=detector_params.get("regex_flags")
                    )
                    
                    # Convert recognizer results to ContentAnalysisResponse format
                    content_results = []
                    for result in recognizer_result_list:
                        if result.score >= score_threshold:
                            # Extract the detected text from the original content using start/end positions
                            detected_text = content[result.start:result.end]
                            
                            # Get recognizer name from metadata if available
                            detector_id = None
                            if result.recognition_metadata:
                                detector_id = result.recognition_metadata.get(
                                    RecognizerResult.RECOGNIZER_NAME_KEY, 
                                    "unknown_recognizer"
                                )
                            
                            content_response = ContentAnalysisResponse(
                                start=result.start,
                                end=result.end,
                                text=detected_text,
                                detection=result.entity_type,
                                detection_type="pii",  # Presidio is primarily for PII
                                score=result.score,
                                detector_id=detector_id,
                                evidence=[],  # Presidio doesn't provide evidence in this format
                                metadata={}
                            )
                            content_results.append(content_response)
                    
                    all_results.append(content_results)

                return Response(
                    json.dumps(all_results, default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)),
                    content_type="application/json",
                )
            except TypeError as te:
                error_msg = (
                    f"Failed to parse /api/v1/text/contents request "
                    f"for AnalyzerEngine.analyze(). {te.args[0]}"
                )
                self.logger.error(error_msg)
                return jsonify(error=error_msg), 400

            except Exception as e:
                self.logger.error(
                    f"A fatal error occurred during execution of "
                    f"AnalyzerEngine.analyze(). {e}"
                )
                return jsonify(error=e.args[0]), 500

        @self.app.route("/analyze", methods=["POST"])
        def analyze() -> Tuple[str, int]:
            """Execute the analyzer function."""
            # Parse the request params
            try:
                req_data = AnalyzerRequest(request.get_json())
                if not req_data.text:
                    raise Exception("No text provided")

                if not req_data.language:
                    raise Exception("No language provided")

                recognizer_result_list = self.engine.analyze(
                    text=req_data.text,
                    language=req_data.language,
                    correlation_id=req_data.correlation_id,
                    score_threshold=req_data.score_threshold,
                    entities=req_data.entities,
                    return_decision_process=req_data.return_decision_process,
                    ad_hoc_recognizers=req_data.ad_hoc_recognizers,
                    context=req_data.context,
                    allow_list=req_data.allow_list,
                    allow_list_match=req_data.allow_list_match,
                    regex_flags=req_data.regex_flags
                )

                return Response(
                    json.dumps(
                        recognizer_result_list,
                        default=lambda o: o.to_dict(),
                        sort_keys=True,
                    ),
                    content_type="application/json",
                )
            except TypeError as te:
                error_msg = (
                    f"Failed to parse /analyze request "
                    f"for AnalyzerEngine.analyze(). {te.args[0]}"
                )
                self.logger.error(error_msg)
                return jsonify(error=error_msg), 400

            except Exception as e:
                self.logger.error(
                    f"A fatal error occurred during execution of "
                    f"AnalyzerEngine.analyze(). {e}"
                )
                return jsonify(error=e.args[0]), 500

        @self.app.route("/recognizers", methods=["GET"])
        def recognizers() -> Tuple[str, int]:
            """Return a list of supported recognizers."""
            language = request.args.get("language")
            try:
                recognizers_list = self.engine.get_recognizers(language)
                names = [o.name for o in recognizers_list]
                return jsonify(names), 200
            except Exception as e:
                self.logger.error(
                    f"A fatal error occurred during execution of "
                    f"AnalyzerEngine.get_recognizers(). {e}"
                )
                return jsonify(error=e.args[0]), 500

        @self.app.route("/supportedentities", methods=["GET"])
        def supported_entities() -> Tuple[str, int]:
            """Return a list of supported entities."""
            language = request.args.get("language")
            try:
                entities_list = self.engine.get_supported_entities(language)
                return jsonify(entities_list), 200
            except Exception as e:
                self.logger.error(
                    f"A fatal error occurred during execution of "
                    f"AnalyzerEngine.supported_entities(). {e}"
                )
                return jsonify(error=e.args[0]), 500

        @self.app.errorhandler(HTTPException)
        def http_exception(e):
            return jsonify(error=e.description), e.code

def create_app(): # noqa
    server = Server()
    return server.app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port)
