"""Provider-independent research agent contracts and orchestration."""

from .contracts import (
    AgentDefinition,
    AgentProposal,
    BackendLocation,
    HypothesisProposal,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    PredictionProposal,
    ResearchAgent,
)
from .coordination import CoordinatorProtocolError, CoordinatorSession
from .examination import ExaminationLimitError, PrivateExamination
from .independent import AgentProtocolError, IndependentPass
from .roles import (
    Coordinator,
    CoordinatorReport,
    ExaminationChallenge,
    ExaminationDisposition,
    ExaminationExchange,
    ExaminationResult,
    Examiner,
    ExaminerReview,
    InvestigationPlan,
    Thinker,
    ThinkerProposal,
    ThinkerResponse,
    ThinkerTask,
)

__all__ = [
    "AgentDefinition",
    "AgentProposal",
    "AgentProtocolError",
    "BackendLocation",
    "Coordinator",
    "CoordinatorProtocolError",
    "CoordinatorReport",
    "CoordinatorSession",
    "ExaminationChallenge",
    "ExaminationDisposition",
    "ExaminationExchange",
    "ExaminationLimitError",
    "ExaminationResult",
    "Examiner",
    "ExaminerReview",
    "HypothesisProposal",
    "IndependentPass",
    "InvestigationPlan",
    "ModelBackend",
    "ModelRequest",
    "ModelResponse",
    "PredictionProposal",
    "PrivateExamination",
    "ResearchAgent",
    "Thinker",
    "ThinkerProposal",
    "ThinkerResponse",
    "ThinkerTask",
]
