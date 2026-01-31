import logging

from utils import eat_prefix, send_request, create_abstract

logger = logging.getLogger("__main__")
OPENALEX_WORK_URL = "https://api.openalex.org/works/{}"

def create_pub_date(pub_year, pub_date, target=True):
    if pub_date is not None:
        return pub_date
    elif pub_year is not None:
        if target:
            return f"{pub_year}-01-01"
        else:
            return f"{pub_year}-12-31"
    else:
        return None

class RelatedAbstracts():
    def __init__(self, 
                 pub_date,
                 alexid, 
                 init_list=None):
        if init_list is None:
            self.buffer = []
        else:
            self.buffer = init_list
    
        self.pub_date = pub_date
        self.alexid = self.validate_id(alexid)
        self.titles = {}
        self.abstracts = {} 
        self.page = 0
        self.max_pages = 10

    def size(self):
        return len(self.titles)
    
    def validate_id(self, alexid):
        alexid = eat_prefix(alexid)
        if alexid.startswith("W"):
            return alexid   
        else:
            logger.warning(f"Invalid OpenAlex ID: {alexid}.")
            return None
        
    def find_topic(self, alexid):
            params = {
                "select": f"primary_topic"   
            }
            data = send_request(
                OPENALEX_WORK_URL.format(alexid),
                params,
                30
            )
            if data is None:
                return None
            print(data)
            exit()
            topic = eat_prefix(data['primary_topic']['id'])
            logger.debug(f"Primary topic for {alexid} is {topic}.")   
            exit()
        
    def find_new_related_works(self):
        if self.alexid is None:
            self.page = self.max_pages + 1
            return None
        topic = self.find_topic(self.alexid)
        ... 

    def _download_title_abstract(self, alexid):
            params = {
                "select": f"publication_year,publication_date,abstract_inverted_index,title"
            }
            data = send_request(
                OPENALEX_WORK_URL.format(alexid),
                params,
                30
            )
            if data is None:
                return None, None
            # first check date 
            work_pub_date = create_pub_date(
                data.get("publication_year"),
                data.get("publication_date"),
                target=False
            )
            if work_pub_date is None:
                logger.warning(f"Missing publication date for {alexid}. Skipping.")
                return None, None       
            if work_pub_date > self.pub_date:
                logger.warning(f"Publication date {work_pub_date} later than target {self.pub_date} for {alexid}. Skipping.")
                return None, None
            
            title = data.get("title")
            abstract_index = data.get("abstract_inverted_index")
            if abstract_index:
                abstract = create_abstract(abstract_index)
            else:
                logger.warning(f"No abstract for {alexid}.")
                abstract = None
            return title, abstract

    def download_data(self):
        for alexid in self.buffer:
            title, abstract = self._download_title_abstract(alexid)
            if title is not None and abstract is not None:
                self.titles[alexid] = title
                self.abstracts[alexid] = abstract
            else:
                logger.warning(f"Missing title or abstract for {alexid}. Skipping.")    

    def populate(self):
        self.download_data()
        logger.debug(f"Size after initial download: {self.size()}.")
        while self.size() < 5 and self.page < self.max_pages:
            self.find_new_related_works()
            self.download_data()