import os

class Config:
    """Base configuration class."""
    DEBUG = False
    TESTING = False
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris.pkl')
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'app.log')

    IRIS_SPECIES_MAP = {
        0: {"name": "Iris-setosa", "description": "Known for small petals and sepals. Native to Japan. Often blue-purple or white in color."},
        1: {"name": "Iris-versicolor", "description": "Often found in North America. Features intermediate petal and sepal sizes. Colors include violet, blue, and reddish-purple."},
        2: {"name": "Iris-virginica", "description": "A larger species native to the eastern United States. Has the largest petals and sepals among the three. Typically purple or blue."},
    }

    IRIS_FEATURE_RANGES = {
        'sepal length (cm)': {'min': 4.3, 'max': 7.9, 'step': 0.1},
        'sepal width (cm)': {'min': 2.0, 'max': 4.4, 'step': 0.1},
        'petal length (cm)': {'min': 1.0, 'max': 6.9, 'step': 0.1},
        'petal width (cm)': {'min': 0.1, 'max': 2.5, 'step': 0.1}
    }


class DevelopmentConfig(Config):
    """Development specific configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production specific configuration."""
    pass

# Dictionary to easily select config based on environment
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}