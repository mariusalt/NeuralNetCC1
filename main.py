from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import traceback
import sys
import csv
import random

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument('--ignore-certificate-errors-spki-list')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(executable_path='/home/mat/Desktop/chromedriver',chrome_options=options)
# driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\chromedriver.exe", chrome_options=chrome_options)
for i in range(1, 100):
    ttt = random.randint(30, 500)
    driver.get('https://www.google.de/')
    print("check1")
    time.sleep(7)
    if i == 1:
        WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe")))
        agree = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="introAgreeButton"]/span/span')))
        agree.click()
    i = +1
    # back to the main page
    driver.switch_to_default_content()
    # driver.find_element_by_xpath("//h3[contains(@class, 'RveJvd snByac')]").click()
    # driver.find_element_by_xpath('//*[@id="introAgreeButton"]').click()
    print("check2")
    suchfeld = driver.find_element_by_name("q")
    print("check3")
    suchfeld.clear()
    print("check4")
    # each bank - search only in appropriate language
    u = "Marius Alt"  # composing the search term
    # uu = u.decode('utf8')
    suchfeld.clear()
    suchfeld.send_keys(u)
    time.sleep(1)
    suchfeld.send_keys('\n')
    time.sleep(4)
    # fld2 = browser.find_element_by_name("btnK");
    # fld2.click()
    print("check5")

    print("check6")
    #   WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[7]/div/div[9]/div[1]/div/div[2]/div[2]/div/div/div/div[5]/div/div[1]/a/h3')))

    time.sleep(5)
    try:
        driver.find_element_by_xpath("//*[contains(text(), 'Marius Alt: Personal Website')]").click()
        # link1=driver.find_element_by_xpath(".//*[text()='Personal Website']")
        print("check7")
        time.sleep(ttt)
    except:
        print("not on first page")
    time.sleep(5)
    try:
        link = driver.find_element_by_xpath(
            "/html/body/div[7]/div/div[9]/div[1]/div/div[6]/span[1]/table/tbody/tr/td[3]/a")
        ref = link.get_attribute("href")
        driver.get(ref)
        time.sleep(5)
        driver.find_element_by_xpath("//*[contains(text(), 'Marius Alt: Personal Website')]").click()
        # link1=driver.find_element_by_xpath(".//*[text()='Personal Website']")
        print("check7")
        time.sleep(ttt)
    except:
        print("not on second page")
    time.sleep(5)
    try:
        link = driver.find_element_by_xpath(
            "/html/body/div[7]/div/div[9]/div[1]/div/div[6]/span[1]/table/tbody/tr/td[4]/a")
        ref = link.get_attribute("href")
        driver.get(ref)
        time.sleep(5)
        driver.find_element_by_xpath("//*[contains(text(), 'Marius Alt: Personal Website')]").click()
        # link1=driver.find_element_by_xpath(".//*[text()='Personal Website']")
        print("check7")
        time.sleep(ttt)
    except:
        print("not on second page")

    # results = browser.find_element_by_xpath("//div[@aria-label='Page4']/div[@class='f1' and text()='Page 4']");
    # for i, fld in enumerate(results):
    #     # print fld.get_attribute('innerHTML')
    #     url = fld.get_attribute('href')
    #     printUrl = url
    #     if len(url) > 110:
    #         printUrl = url[:110]
    #     print(" url %d is %s" % (i, printUrl))
    #     outRow = ['col1', bank[1], "%s" % i]
    #     outRow.extend([url])
    #     # outRow.pop() # remove last element
    #     outLine = ";".join(outRow) + "\n"
    #     outFile.write(outLine)
    #     # time.sleep(1)
    # time.sleep(2)
    # # next input

    print("regular finish")

    # # Find an element merely by its text content
    # def nodesByText(browser, needle):
    #     expr = "//*[contains(text(),'%s')]" % needle
    #     listOfNodes = browser.find_elements_by_xpath(expr)
    #     for i, nd in enumerate(listOfNodes):
    #         print("fld %d name is %10s %10s" % (i, nd.get_attribute('name'), nd.get_attribute('id')))
    #         nd.click() # Just for fun

    #     #fld2 = driver.find_element_by_class_name("fc-button.fc-button-consent")
    # #    element = WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[2]/section[1]/div[3]/button[1]")))

    #     #   element.click();

    #     # fdl2 = driver.find_element_by_link_text("I ACCEPT")
    #     #fld2 = driver.find_element_by_class_name(xpath)
    #     # fld2=driver.find_element_by_css_selector("[role=button]")
    #     #  fld2.click()
    #     # time.sleep(3)
    #     if "noticed some unusual activity coming from your computer network" in driver.page_source:
    #         print("CAPTCHA encountered")
    #         quit()
    #     #  suchfeld = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//a[@class="form-control"]')))
    #     suchfeld=driver.find_element_by_xpath("/html/body/div[1]/header/nav[2]/div/div[1]/div[2]/div[2]/div/form/div/input")
    #     suchfeld.clear()
    #     #  suchfeld.send_keys(tel[0])
    #     suchfeld.send_keys(tel[0])
    #     time.sleep(1)
    #     suchfeld.send_keys('\n')
    #     time.sleep(3)
    #     link=driver.find_element_by_xpath('//*[@id="search-phones"]/div[3]/div[1]/div/a')
    #     ref=link.get_attribute("href")
    #     print(ref)
    #     driver.get(ref)
    #     time.sleep(2)
    #     #  b=[]
    #     battery=driver.find_element_by_xpath('//*[@id="phones-content-inner"]/div/div/div/div[1]/article/div/section[1]/div/aside/ul[1]/li[6]/span[2]').text
    #     print(battery)
    #     bat.append(battery)
    # #    print(b)
    #     for n,l in enumerate(bat):
    #         s = str(l)
    #         s = s[:4]
    #         bat[n]=s
    #     # print(b)
    #     # bat.append(b)
    #     name=driver.find_element_by_xpath('//*[@id="phones-content-inner"]/div/div/div/div[1]/article/div/section[1]/div/header/div[1]/div/div[1]/h1').text
    #     print(name)
    # #    m=[]
    #     #   m.append(name)
    #     nam.append(name)

    #     telo.append(tel[0])

    #     if name==tel[0]:
    #         comp.append('same')
    #     else:
    #         comp.append('diff')

    #     with open('battery.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['TelHom','Battery','TelList','Match'])
    #         writer.writerows(zip(nam,bat,telo,comp))

    #     #       browser.get(landtag[1])
    # #        time.sleep(2)
    #     #       if "noticed some unusual activity coming from your computer network" in browser.page_source:
    #     #          print("CAPTCHA encountered")
    #     #         sendEmail("selenium failure: captcha",
    #     #    "captcha", "buchmann@zew.de")
    #     #           quit()
    #     #      # Search for single element
    #     #     direkt = browser.find_element_by_xpath(
    #     #    "//table[@class='ohne-rand m-u-0']//td//a").get_attribute('href')
    #         #   print(direkt)
    # except Exception as e:
    #     print("\nException in inner loop:\n")
    #     print(traceback.format_exc())

    print("loop end")

driver.quit()
