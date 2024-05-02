from pathlib import Path
from dotenv import load_dotenv



BASE_DIR = Path(__file__).parents[1]


DB_INIT_FILE = BASE_DIR / 'database.ini'
dotenv_path = BASE_DIR / '.example.env'


load_dotenv(dotenv_path)
# load_dotenv()

