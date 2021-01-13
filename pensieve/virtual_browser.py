import argparse
import json
import os
import signal
import subprocess
import sys
from time import sleep
from urllib.request import urlopen

from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

# TO RUN: download https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz
# run sudo apt-get install python-setuptools
# run sudo apt-get install xvfb
# after untar, run sudo python setup.py install
# follow directions here: https://pypi.python.org/pypi/PyVirtualDisplay to install pyvirtualdisplay

# For chrome, need chrome driver: https://code.google.com/p/selenium/wiki/ChromeDriver
# chromedriver variable should be path to the chromedriver
# the default location for firefox is /usr/bin/firefox and chrome binary is /usr/bin/google-chrome
# if they are at those locations, don't need to specify


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Virtual Browser")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')

    parser.add_argument('--ip', type=str, help='IP address.')
    parser.add_argument('--port', type=int, help='Port number.')
    parser.add_argument('--abr', type=str, help='ABR algorithm.')
    # parser.add_argument('--trace_file', type=str, help='Path to trace file.')
    parser.add_argument('--run_time', type=int, default=320,
                        help="Running time.")
    parser.add_argument('--sleep_time', type=int,
                        default=2, help="Sleep time.")

    return parser.parse_args()


def timeout_handler(signum, frame):
    raise Exception("Timeout")


def main():
    args = parse_args()
    ip = args.ip
    port_number = args.port
    abr_algo = args.abr
    run_time = args.run_time
    # process_id = sys.argv[4]
    # trace_file = args.trace_file
    sleep_time = args.sleep_time

    # prevent multiple process from being synchronized
    sleep(int(sleep_time))

    # generate url
    url = 'http://{}:{}/myindex_{}.html'.format(ip, port_number, abr_algo)
    # ip = json.loads(urlopen("http://ip.jsontest.com/").read().decode('utf-8'))['ip']
    # url = 'http://{}/myindex_{}.html'.format(ip, abr_algo)
    print(url)

    # timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(run_time + 30)
    display = None
    driver = None
    try:
        # copy over the chrome user dir
        default_chrome_user_dir = '../abr_browser_dir/chrome_data_dir'
        # chrome_user_dir = '/tmp/chrome_user_dir_id_' + process_id
        chrome_user_dir = '/tmp/chrome_user_dir_id'  # + process_id
        os.system('rm -r ' + chrome_user_dir)
        os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)

        # start abr algorithm server
        # if abr_algo == 'RL':
        #     command = 'exec /usr/bin/python ../rl_server/rl_server_no_training.py ' + trace_file
        # elif abr_algo == 'fastMPC':
        #     command = 'exec /usr/bin/python ../rl_server/mpc_server.py ' + trace_file
        # elif abr_algo == 'robustMPC':
        #     command = 'exec /usr/bin/python ../rl_server/robust_mpc_server.py ' + trace_file
        # else:
        #     command = 'exec /usr/bin/python ../rl_server/simple_server.py ' + \
        #         abr_algo + ' ' + trace_file
        #
        # proc = subprocess.Popen(command, stdout=subprocess.PIPE,
        #                         stderr=subprocess.PIPE, shell=True)
        sleep(2)

        # to not display the page in browser
        display = Display(visible=False, size=(800, 600))
        display.start()

        # initialize chrome driver
        options = Options()
        chrome_driver = '../abr_browser_dir/chromedriver'
        options.add_argument('--user-data-dir=' + chrome_user_dir)
        options.add_argument('--ignore-certificate-errors')
        driver = webdriver.Chrome(chrome_driver, chrome_options=options)

        # run chrome
        driver.set_page_load_timeout(10)
        driver.get(url)

        print()
        sleep(run_time)
        driver.quit()
        display.stop()

        # kill abr algorithm server
        # proc.send_signal(signal.SIGINT)
        # proc.kill()

        print('done')

    except Exception as e:
        if display is not None:
            display.stop()
        if driver is not None:
            driver.quit()
        # try:
        #     proc.send_signal(signal.SIGINT)
        # except:
        #     pass

        print(e)


if __name__ == '__main__':
    main()
