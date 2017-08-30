import sys
from boilerpipe.extract import Extractor

if sys.version_info.major==2:
    import urllib2
else:
    import urllib as urllib2

def ensure_valid(url):
    if not url[:7] in ['http://',] and not url[:8] in ['https://',]:
        url='http://'+url
    return url

def extract_main_text(html_text):
    extractor=Extractor(extractor='ArticleExtractor',html=html_text)
    extracted_text=extractor.getText()
    return extracted_text

def get_content_from_url(url):
    url=ensure_valid(url)
    response=urllib2.urlopen(url)
    extracted_text=response.read()
    return extract_main_text(extracted_text)

if __name__=='__main__':
    if len(sys.argv)!=2:
        print('Usage: python html.py <url>')
        exit(0)

    url=ensure_valid(sys.argv[1])
    response=urllib2.urlopen(url)
    extracted_text=response.read()
    print(extract_main_text(extracted_text))
