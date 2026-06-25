from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
class wea:


     def suan(self,cit):
      try:
        driver_path = r"C:\Users\57422\Desktop\小发明\网络爬虫\edgedriver_win64\msedgedriver.exe"
        options = Options()

        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-insecure-localhost")
        options.add_argument("--unsafely-treat-insecure-origin-as-secure=https://weather.cma.cn")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
          # 启用无头模式
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
          # 设置窗口大小（很重要！避免元素不可见）
        options.add_argument("--window-size=1920,1080")
          # 设置用户代理，让网站认为是正常浏览器
        options.add_argument(
              "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        service = Service(executable_path=driver_path)
        driver = webdriver.Edge(service=service, options=options)

          # 打开网页
        driver.get("https://weather.cma.cn/")
        time.sleep(5)  # 多等一会
    # 找到搜索框
        search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
    # 输入城市
        search_box.clear()
        search_box.send_keys(cit)
        time.sleep(2)
    # 按回车键搜索
        search_box.send_keys(Keys.ENTER)
        # 智能等待温度元素出现（最多等15秒）
        driver.refresh()
        wait = WebDriverWait(driver, 25)
        temperature_element = wait.until(
          EC.presence_of_element_located((By.ID, "temperature"))
        )
    # 直接在当前页面提取温度（不需要重新访问URL）
        tem = temperature_element.text.strip()
        tianqi =driver.find_element(By.CSS_SELECTOR,".pull-left.day.actived .day-item:nth-child(3)")

        tq = tianqi.text.strip()
        return f"{cit}天气为{tq},{tem}"
    
      except Exception as e:
         print(f"错误: {str(e)}")

