from selenium import webdriver
import time

from bs4 import BeautifulSoup


class ProgrammableWebScrapy:

    def __init__(self):
        self.browser = webdriver.Firefox(executable_path="D:\ProjectFolder\geckodriver.exe")

    def __del__(self):
        self.browser.close()

    def get_page(self, url):
        # get page from browser
        self.browser.get(url)
        page_source = self.browser.page_source

        return page_source

    def parse_html(self, page_source):

        a_list = []

        html = BeautifulSoup(page_source, "html5lib")

        trs = html.find_all("tr", class_=["odd", "even"])
        # print("tr len: %d" % len(trs))

        for tr in trs:
            a_link = tr.find_all("a")[0]
            new_url = "https://www.programmableweb.com" + a_link["href"]
            # print("Process:  " + a_link.text + "  , url: " + a_link["href"])
            a_list.append(new_url)
        return a_list

    def parse_single_html(self, page_source):

        html = BeautifulSoup(page_source, 'html5lib')

        # Title
        title_div = html.find('div', class_='node-header')
        try:

            title = title_div.find('h1').text
            # print('Title: ' + title)

            # Tags


            # description
            description_div = html.find('div', {'id': 'tabs-header-content'})
            description = description_div.text.strip()
            # print('Description: ' + repr(description))

            # table contents
            myTabContent_div = html.find('div', {'id': 'myTabContent'})

            field_divs = myTabContent_div.find_all('div', {'class': 'field'})
            # print('field div len: %d ' % len(field_divs))
            myTabContentStr = ''
            for field_div in field_divs:
                label = field_div.find('label').text
                span = field_div.find('span')
                if 'API Endpoint' in label:
                    myTabContentStr += '"APIEndpoint":"' + span.text + '",'
                elif 'API Portal / Home Page' in label:
                    myTabContentStr += '"APIPortalHomePage":"' + span.text + '",'
                elif 'Primary Category' in label:
                    myTabContentStr += '"PrimaryCategory":"' + span.text + '",'
                elif 'Secondary Categories' in label:
                    myTabContentStr += '"SecondaryCategories":"' + span.text + '",'
                elif 'API Provider' in label:
                    myTabContentStr += '"APIProvider":"' + span.text + '",'
                elif 'SSL Support' in label:
                    myTabContentStr += '"SSLSupport":"' + span.text + '",'
                elif 'API Forum / Message Boards' in label:
                    myTabContentStr += '"APIForumMessageBoards":"' + span.text + '",'
                elif 'Twitter URLr' in label:
                    myTabContentStr += '"TwitterURL":"' + span.text + '",'
                elif 'Support Email Address' in label:
                    myTabContentStr += '"SupportEmailAddress":"' + span.text + '",'
                elif 'Interactive Console URLr' in label:
                    myTabContentStr += '"InteractiveConsoleURL":"' + span.text + '",'
                elif 'Authentication Model' in label:
                    myTabContentStr += '"AuthenticationModel":"' + span.text + '",'
                elif 'Is the API Design/Description Non-Proprietary' in label:
                    myTabContentStr += '"IstheAPIDesign":"' + span.text + '",'
                elif 'Scope' in label:
                    myTabContentStr += '"Scope":"' + span.text + '",'
                elif 'Device Specific' in label:
                    myTabContentStr += '"DeviceSpecific":"' + span.text + '",'
                elif 'Docs Home Page UR' in label:
                    myTabContentStr += '"DocsHomePageUR":"' + span.text + '",'
                elif 'Architectural Style' in label:
                    myTabContentStr += '"ArchitecturalStyle":"' + span.text + '",'
                elif 'Supported Request Formats' in label:
                    myTabContentStr += '"SupportedRequestFormats":"' + span.text + '",'
                elif 'Supported Response Formats' in label:
                    myTabContentStr += '"SupportedResponseFormats":"' + span.text + '",'
                elif 'Is This an Unofficial API' in label:
                    myTabContentStr += '"IsThisanUnofficialAPI":"' + span.text + '",'
                elif 'Restricted Access' in label:
                    myTabContentStr += '"RestrictedAccess":"' + span.text + '",'

            myTabContentStr = myTabContentStr[:-1]

            result = '"Title":"' + title.strip() + '","Description":"' + description.strip().replace('\n\n',
                                                                                                     ' ') + '",' + myTabContentStr.strip()
            return result
        except:
            return ''


if __name__ == '__main__':
    basic_url = 'https://www.programmableweb.com/category/all/apis?page='
    total_pages = 695

    programWeb = ProgrammableWebScrapy()

    all_a_list = []

    with open("ProgrammWebScrapy.txt", "a") as w:
        # w.write("[\n")

        for i in range(690, total_pages):
            print("Page: %d" % i)
            url = basic_url + str(i)

            page_source = programWeb.get_page(url)

            a_list = programWeb.parse_html(page_source)

            for a_item in a_list:
                try:
                    page_source = programWeb.get_page(a_item)

                    json_item = programWeb.parse_single_html(page_source)
                    if json_item == '':
                        continue
                    json_item = "{" + json_item + "},\n"
                    w.write(json_item)
                except:
                    continue
                time.sleep(2)
            time.sleep(3)
        w.write("]\n")
    # print("All a list len: %d" % len(all_a_list))

    # json_data = []

    # with open("ProgrammWebScrapy.txt", "w") as w:
    #     w.write("[\n")
    #
    #     for a_item in all_a_list:
    #         page_source = programWeb.get_page(a_item)
    #
    #         json_item = programWeb.parse_single_html(page_source)
    #         json_item = "{" + json_item + "},\n"
    #
    #         w.write(json_item)
    #         time.sleep(1)
    #
    #     w.write("]\n")

    # for a_item in all_a_list:
    #     page_source = programWeb.get_page(a_item)
    #
    #     json_item = programWeb.parse_single_html(page_source)
    #     json_item = "{" + json_item + "}"
    #     json_data.append(json_item)
    #
    #     time.sleep(3)
    #
    # with open("ProgrammWebScrapy.txt", "w") as w:
    #
    #     for idx, json_item in enumerate(json_data):
    #         if idx == 0:
    #             json_item = "[" + json_item + ","
    #         elif idx == len(json_data) - 1:
    #             json_item = json_data + "]"
    #         else:
    #             json_item = json_item + ","
    #         w.write(json_item)



