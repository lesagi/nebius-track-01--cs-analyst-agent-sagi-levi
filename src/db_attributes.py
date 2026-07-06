from enum import Enum


class Flags(str, Enum):
    M = "M"
    L = "L"
    B = "B"
    I = "I"
    C = "C"
    N = "N"
    P = "P"
    Q = "Q"
    W = "W"
    K = "K"
    E = "E"
    Z = "Z"
    D = "D"
    G = "G"
    R = "R"
    Y = "Y"


FLAG_DESCRIPTIONS: dict[Flags, str] = {
    Flags.M: 'Morphological variation: inflectional and derivational (e.g., "is my SIM card active", "is my SIM card activated")',
    Flags.L: 'Semantic variations: synonyms, use of hyphens, compounding (e.g., "what’s my billing date", "what’s my anniversary date")',
    Flags.B: 'Basic syntactic structure: (e.g., "activate my SIM card", "I need to activate my SIM card")',
    Flags.I: 'Interrogative structure: (e.g., "can you activate my SIM card?", "how do I activate my SIM card?")',
    Flags.C: 'Coordinated syntactic structure: (e.g., "I have a new SIM card, what do I need to do to activate it?")',
    Flags.N: 'Negation: (e.g., "I do not want this item, where to cancel my order?")',
    Flags.P: 'Politeness variation: (e.g., "could you help me activate my SIM card, please?")',
    Flags.Q: 'Colloquial variation: (e.g., "can u activ8 my SIM?")',
    Flags.W: 'Offensive language: (e.g., "I want to talk to a f*&%*g agent")',
    Flags.K: 'Keyword mode: (e.g., "activate SIM", "new SIM")',
    Flags.E: 'Use of abbreviations: (e.g., "I\'m / I am interested in getting a new SIM")',
    Flags.Z: 'Errors and Typos: spelling issues, wrong punctuation (e.g., "how can i activaet my card")',
    Flags.D: 'Indirect speech: (e.g., "ask my agent to activate my SIM card")',
    Flags.G: 'Regional variations: US English vs UK English ("truck" vs "lorry"), France French vs Canadian French ("tchatter" vs "clavarder")',
    Flags.R: 'Respect structures: Language-dependent variations (English: "may" vs "can…", French: "tu" vs "vous...", Spanish: "tú" vs "usted...")',
    Flags.Y: 'Code switching: (e.g., "activer ma SIM card")',
}


class Category(str, Enum):
    ACCOUNT = "ACCOUNT"
    CANCEL = "CANCEL"
    CONTACT = "CONTACT"
    DELIVERY = "DELIVERY"
    FEEDBACK = "FEEDBACK"
    INVOICE = "INVOICE"
    ORDER = "ORDER"
    PAYMENT = "PAYMENT"
    REFUND = "REFUND"
    SHIPPING = "SHIPPING"
    SUBSCRIPTION = "SUBSCRIPTION"


class IntentType(str, Enum):
    CREATE_ACCOUNT = "create_account"
    DELETE_ACCOUNT = "delete_account"
    EDIT_ACCOUNT = "edit_account"
    RECOVER_PASSWORD = "recover_password"
    REGISTRATION_PROBLEMS = "registration_problems"
    SWITCH_ACCOUNT = "switch_account"
    CHECK_CANCELLATION_FEE = "check_cancellation_fee"
    CONTACT_CUSTOMER_SERVICE = "contact_customer_service"
    CONTACT_HUMAN_AGENT = "contact_human_agent"
    DELIVERY_OPTIONS = "delivery_options"
    DELIVERY_PERIOD = "delivery_period"
    COMPLAINT = "complaint"
    REVIEW = "review"
    CHECK_INVOICE = "check_invoice"
    GET_INVOICE = "get_invoice"
    CANCEL_ORDER = "cancel_order"
    CHANGE_ORDER = "change_order"
    PLACE_ORDER = "place_order"
    TRACK_ORDER = "track_order"
    CHECK_PAYMENT_METHODS = "check_payment_methods"
    PAYMENT_ISSUE = "payment_issue"
    CHECK_REFUND_POLICY = "check_refund_policy"
    GET_REFUND = "get_refund"
    TRACK_REFUND = "track_refund"
    CHANGE_SHIPPING_ADDRESS = "change_shipping_address"
    SET_UP_SHIPPING_ADDRESS = "set_up_shipping_address"
    NEWSLETTER_SUBSCRIPTION = "newsletter_subscription"


CATEGORIES_AND_INTENTS = [
    {
        "category": Category.ACCOUNT,
        "intent_types": [
            IntentType.CREATE_ACCOUNT,
            IntentType.DELETE_ACCOUNT,
            IntentType.EDIT_ACCOUNT,
            IntentType.RECOVER_PASSWORD,
            IntentType.REGISTRATION_PROBLEMS,
            IntentType.SWITCH_ACCOUNT,
        ],
    },
    {
        "category": Category.CANCEL,
        "intent_types": [
            IntentType.CHECK_CANCELLATION_FEE,
        ],
    },
    {
        "category": Category.CONTACT,
        "intent_types": [
            IntentType.CONTACT_CUSTOMER_SERVICE,
            IntentType.CONTACT_HUMAN_AGENT,
        ],
    },
    {
        "category": Category.DELIVERY,
        "intent_types": [
            IntentType.DELIVERY_OPTIONS,
            IntentType.DELIVERY_PERIOD,
        ],
    },
    {
        "category": Category.FEEDBACK,
        "intent_types": [
            IntentType.COMPLAINT,
            IntentType.REVIEW,
        ],
    },
    {
        "category": Category.INVOICE,
        "intent_types": [
            IntentType.CHECK_INVOICE,
            IntentType.GET_INVOICE,
        ],
    },
    {
        "category": Category.ORDER,
        "intent_types": [
            IntentType.CANCEL_ORDER,
            IntentType.CHANGE_ORDER,
            IntentType.PLACE_ORDER,
            IntentType.TRACK_ORDER,
        ],
    },
    {
        "category": Category.PAYMENT,
        "intent_types": [
            IntentType.CHECK_PAYMENT_METHODS,
            IntentType.PAYMENT_ISSUE,
        ],
    },
    {
        "category": Category.REFUND,
        "intent_types": [
            IntentType.CHECK_REFUND_POLICY,
            IntentType.GET_REFUND,
            IntentType.TRACK_REFUND,
        ],
    },
    {
        "category": Category.SHIPPING,
        "intent_types": [
            IntentType.CHANGE_SHIPPING_ADDRESS,
            IntentType.SET_UP_SHIPPING_ADDRESS,
        ],
    },
    {
        "category": Category.SUBSCRIPTION,
        "intent_types": [
            IntentType.NEWSLETTER_SUBSCRIPTION,
        ],
    },
]
