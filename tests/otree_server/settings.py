from os import environ


SESSION_CONFIGS = [
    dict(
        name='ertan2009',
        display_name="Ertan et al. 2009 — Who to Punish?",
        app_sequence=['ertan2009'],
        num_demo_participants=1,
    ),
    dict(
        name='public_goods',
        display_name="Public Goods",
        app_sequence=['public_goods_simple'],
        num_demo_participants=3,
    ),
    dict(
        name='dictator',
        display_name="Dictator Game",
        app_sequence=['dictator'],
        num_demo_participants=2,
    ),
    dict(
        name='prisoner',
        display_name="Prisoner's Dilemma",
        app_sequence=['prisoner'],
        num_demo_participants=2,
    ),
    dict(
        name='trust',
        display_name="Trust Game",
        app_sequence=['trust_simple'],
        num_demo_participants=2,
    ),
    dict(
        name='guess_two_thirds',
        display_name="Guess 2/3 of the Average",
        app_sequence=['guess_two_thirds', 'payment_info'],
        num_demo_participants=3,
    ),
    dict(
        name='survey',
        app_sequence=['survey', 'payment_info'],
        num_demo_participants=1,
    ),
]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ROOMS = [
    dict(
        name='econ101',
        display_name='Econ 101 class',
        participant_label_file='_rooms/econ101.txt',
    ),
    dict(name='live_demo', display_name='Room for live demo (no participant labels)'),
]

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'admin')

OTREE_REST_KEY = environ.get('OTREE_REST_KEY', 'test-rest-key')

DEMO_PAGE_INTRO_HTML = """
Here are some oTree games.
"""


SECRET_KEY = '8137439220110'

INSTALLED_APPS = ['otree']
