from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, InvalidSessionIdException, WebDriverException
import time
import csv
import os
import glob
from datetime import datetime

# --- Config ---
LOGIN_URL = "https://4rabet365.com/"  # homepage with login button
USERNAME = "9852241667"
PASSWORD = "Target@2025"

def create_driver():
    """Create and return a new Chrome WebDriver instance."""
    options = Options()
    # options.add_argument('--headless')  # Uncomment to run headless
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)

def login_and_navigate_to_aviator(driver, wait):
    """Perform login and navigate to Aviator page. Returns True if successful."""
    try:
        print("🌐 Step 1: Opening website...")
        driver.get(LOGIN_URL)
        time.sleep(2)
        
        # Close any popup if present
        print("🔍 Step 2: Checking for popups...")
        try:
            close_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.close-btn")))
            close_button.click()
            print("✅ Popup closed")
            time.sleep(1)
        except TimeoutException:
            print("ℹ️  No popup found, continuing...")

        # Click Log In button
        print("🔍 Step 3: Looking for Login button...")
        login_button = wait.until(EC.presence_of_element_located((By.ID, "auth_btn")))
        print("✅ Login button found")
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", login_button)
        time.sleep(1)
        
        try:
            wait.until(EC.element_to_be_clickable(login_button))
            login_button.click()
            print("✅ Login button clicked (regular click)")
        except Exception as e:
            print(f"   Regular click failed ({type(e).__name__}), using JavaScript click...")
            driver.execute_script("arguments[0].click();", login_button)
            print("✅ Login button clicked (JavaScript click)")
        
        time.sleep(3)

        # Try to click "Provide" button (optional)
        print("🔍 Step 4: Checking for 'Provide' button (optional)...")
        short_wait = WebDriverWait(driver, 3)
        try:
            provide_btn = short_wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Provide')]")))
            provide_btn.click()
            print("✅ Provide button clicked")
            time.sleep(2)
        except TimeoutException:
            print("ℹ️  Provide button not found - this may be normal, proceeding to login form...")
            time.sleep(1)

        # Wait for login form
        print("🔍 Step 5: Waiting for login form to appear...")
        try:
            short_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
            print("✅ Login modal visible")
        except TimeoutException:
            print("ℹ️  Modal selector not found, proceeding to look for input fields directly...")
        
        time.sleep(2)

        # Find phone input
        print("🔍 Step 6: Looking for phone input field...")
        phone_input = None
        try:
            phone_input = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[data-auto-test-el="enterPhone"]')))
            print("✅ Phone input found")
        except TimeoutException:
            try:
                print("   Trying alternative phone input selectors...")
                phone_input = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[type="tel"], input[placeholder*="phone"], input[placeholder*="Phone"]')))
                print("✅ Phone input found (alternative selector)")
            except TimeoutException:
                raise Exception("Could not find phone input field.")

        driver.execute_script("arguments[0].scrollIntoView(true);", phone_input)
        time.sleep(0.5)

        # Enter phone number
        print("📝 Step 7: Entering phone number...")
        driver.execute_script("""
        let input = arguments[0];
        input.value = arguments[1];
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        """, phone_input, USERNAME)
        print(f"✅ Phone number entered: {USERNAME}")
        time.sleep(2)
        
        # Find password input
        print("🔍 Step 8: Looking for password input field...")
        password_input = None
        try:
            password_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[data-auto-test-el="enterPassword"]')))
            print("✅ Password input found (data-auto-test-el)")
        except TimeoutException:
            try:
                print("   Trying input[type='password'] selector...")
                password_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="password"]')))
                print("✅ Password input found (type=password)")
            except TimeoutException:
                try:
                    print("   Trying to find password field by label...")
                    password_input = wait.until(EC.presence_of_element_located((By.XPATH, "//label[contains(text(), 'Password')]/following::input[@type='password'] | //input[@type='password' and @data-auto-test-el='enterPassword']")))
                    print("✅ Password input found (by label)")
                except TimeoutException:
                    raise Exception("Could not find password input field.")
        
        if not password_input.is_displayed():
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", password_input)
            time.sleep(0.5)
        print("✅ Password field is ready")
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", password_input)
        time.sleep(0.5)
        
        try:
            password_input.click()
            password_input.clear()
            print("✅ Password field focused and cleared")
        except Exception as e:
            print(f"   Note: Could not click password field directly ({type(e).__name__}), using JavaScript focus...")
            driver.execute_script("arguments[0].focus();", password_input)
            driver.execute_script("arguments[0].value = '';", password_input)
        
        time.sleep(0.5)
        
        # Enter password
        print("📝 Step 9: Entering password...")
        try:
            password_input.send_keys(PASSWORD)
            print("✅ Password entered via send_keys")
        except Exception as e:
            print(f"   send_keys failed ({type(e).__name__}), trying JavaScript...")
            driver.execute_script("""
            let input = arguments[0];
            input.focus();
            input.value = arguments[1];
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            input.dispatchEvent(new Event('blur', { bubbles: true }));
            """, password_input, PASSWORD)
            print("✅ Password entered via JavaScript")
        
        entered_value = driver.execute_script("return arguments[0].value;", password_input)
        if entered_value:
            print(f"✅ Password field verified - contains {len(entered_value)} characters")
        else:
            print("⚠️  Warning: Password field appears empty, trying one more time...")
            driver.execute_script("""
            let input = arguments[0];
            input.focus();
            input.value = arguments[1];
            input.dispatchEvent(new Event('focus', { bubbles: true }));
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            input.dispatchEvent(new Event('blur', { bubbles: true }));
            """, password_input, PASSWORD)
        
        time.sleep(1)

        # Find and click login button
        print("🔍 Step 10: Looking for Sign In button...")
        login_btn = None
        try:
            login_btn = wait.until(EC.presence_of_element_located((By.ID, "login-sign-in-btn")))
            print("✅ Sign In button found (login-sign-in-btn ID)")
        except TimeoutException:
            try:
                login_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-auto-test-el="signIn"]')))
                print("✅ Sign In button found (data-auto-test-el)")
            except TimeoutException:
                try:
                    print("   Trying button by wrapper...")
                    wrapper = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-auto-test-el="auth-dialog-submit-wrapper"]')))
                    login_btn = wrapper.find_element(By.CSS_SELECTOR, 'button[data-auto-test-el="signIn"]')
                    print("✅ Sign In button found (via wrapper)")
                except TimeoutException:
                    try:
                        print("   Trying button by text 'Log In'...")
                        login_btn = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Log In') or contains(text(), 'Sign In')]")))
                        print("✅ Sign In button found (by text)")
                    except TimeoutException:
                        raise Exception("Could not find Sign In button")
        
        print(f"   Button found: {login_btn.tag_name}, ID: {login_btn.get_attribute('id')}, Enabled: {login_btn.is_enabled()}")
        
        if not login_btn.is_enabled():
            print("⚠️  Button is disabled, waiting for it to become enabled...")
            try:
                wait.until(lambda d: login_btn.is_enabled())
                print("✅ Button is now enabled")
            except TimeoutException:
                print("⚠️  Button still disabled after wait - checking form validation...")
        
        # Verify form fields
        print("🔍 Verifying form fields are filled...")
        phone_value = driver.execute_script("return arguments[0].value;", phone_input)
        password_value = driver.execute_script("return arguments[0].value;", password_input)
        
        if not phone_value:
            print("⚠️  Phone field is empty, re-entering...")
            driver.execute_script("""
            let input = arguments[0];
            input.value = arguments[1];
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            """, phone_input, USERNAME)
            time.sleep(0.5)
        
        if not password_value:
            print("⚠️  Password field is empty, re-entering...")
            driver.execute_script("""
            let input = arguments[0];
            input.focus();
            input.value = arguments[1];
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            """, password_input, PASSWORD)
            time.sleep(0.5)
        
        print(f"✅ Phone: {len(phone_value) if phone_value else 0} chars, Password: {len(password_value) if password_value else 0} chars")
        time.sleep(1)
        
        try:
            wait.until(EC.element_to_be_clickable(login_btn))
            print("✅ Sign In button is clickable")
        except TimeoutException:
            print("⚠️  Button not clickable via standard check, will try JavaScript click...")
        
        try:
            wrapper = driver.find_element(By.CSS_SELECTOR, '[data-auto-test-el="auth-dialog-submit-wrapper"]')
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", wrapper)
        except:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", login_btn)
        time.sleep(1)
        
        # Try multiple click strategies
        clicked = False
        
        try:
            print("   Attempting click on button content span...")
            content_span = login_btn.find_element(By.CSS_SELECTOR, ".v-btn__content")
            content_span.click()
            clicked = True
            print("✅ Sign in button clicked successfully (via content span)!")
        except Exception as e:
            print(f"   Content span click failed ({type(e).__name__})")
        
        if not clicked:
            try:
                print("   Attempting regular click...")
                login_btn.click()
                clicked = True
                print("✅ Sign in button clicked successfully (regular click)!")
            except Exception as e:
                print(f"   Regular click failed ({type(e).__name__})")
        
        if not clicked:
            try:
                print("   Attempting JavaScript click with events...")
                driver.execute_script("""
                let btn = arguments[0];
                let overlays = btn.querySelectorAll('.v-btn__overlay, .v-btn__underlay');
                overlays.forEach(ov => ov.style.pointerEvents = 'none');
                btn.focus();
                btn.click();
                """, login_btn)
                clicked = True
                print("✅ Sign in button clicked successfully (JavaScript click)!")
            except Exception as e2:
                print(f"   JavaScript click failed ({type(e2).__name__})")
        
        if not clicked:
            print("❌ All click methods failed!")
            return False
        
        print("✅ Sign in button click completed!")
        time.sleep(2)
        
        # Wait for login to complete
        print("⏳ Waiting for login to complete...")
        time.sleep(5)
        
        try:
            WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
            print("✅ Login modal closed - login appears successful!")
        except TimeoutException:
            print("⚠️  Login modal still visible after 10 seconds")
        
        time.sleep(2)
        
        # Wait for dashboard
        print("🔍 Step 11: Waiting for dashboard to load...")
        try:
            WebDriverWait(driver, 15).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
            print("✅ Login modal closed - dashboard should be loading...")
        except TimeoutException:
            print("⚠️  Login modal still visible, but proceeding to check dashboard...")
        
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body, main, .dashboard, .main-content, [class*='dashboard'], [class*='main']")))
            print("✅ Dashboard elements detected")
        except TimeoutException:
            print("⚠️  Dashboard elements not found, but proceeding...")
        
        print("✅ Dashboard loaded successfully!")
        
        # Navigate to Aviator
        print("🔍 Step 12: Looking for Aviator button...")
        driver.switch_to.default_content()

        def find_aviator():
            try:
                anchor = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href="/casino/slot/aviator"], a[href*="aviator"]')))
                return anchor
            except TimeoutException:
                div = wait.until(EC.presence_of_element_located((By.ID, "top-section__aviator")))
                try:
                    return div.find_element(By.XPATH, "./ancestor::a[1]")
                except Exception:
                    return div

        aviator_clicked = False
        for attempt in range(3):
            try:
                target_clickable = find_aviator()
            except TimeoutException:
                print(f"❌ Aviator element not found on attempt {attempt + 1}")
                continue

            if not target_clickable:
                print("❌ Aviator element could not be located.")
                break

            print(f"✅ Aviator element found (attempt {attempt + 1})")

            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", target_clickable)
                time.sleep(0.5)
                wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='top-section__aviator'] | //a[contains(@href,'aviator')]")))
                target_clickable.click()
                aviator_clicked = True
                print("✅ Aviator clicked successfully (regular click)")
                break
            except StaleElementReferenceException:
                print(f"   ⚠️ Stale element on attempt {attempt + 1}, retrying...")
                continue
            except Exception as e:
                print(f"   Regular click failed ({type(e).__name__}), trying JavaScript click...")
                try:
                    target_clickable = find_aviator()
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_clickable)
                    time.sleep(0.2)
                    driver.execute_script("arguments[0].click();", target_clickable)
                    aviator_clicked = True
                    print("✅ Aviator clicked successfully (JavaScript click)")
                    break
                except Exception as e2:
                    print(f"   JavaScript click failed ({type(e2).__name__}) on attempt {attempt + 1}")
                    continue

        if not aviator_clicked:
            print("❌ Aviator element could not be clicked after retries.")
            return False
        
        time.sleep(60)  # Wait for Aviator page to fully load
        
        # Switch to payouts frame if needed
        def switch_to_payouts_frame():
            driver.switch_to.default_content()
            frames = driver.find_elements(By.TAG_NAME, "iframe")
            for idx, frame in enumerate(frames):
                try:
                    driver.switch_to.frame(frame)
                    if driver.find_elements(By.CSS_SELECTOR, ".payouts-wrapper, .payouts-block .payout"):
                        print(f"✅ Switched to iframe {idx} containing payouts UI")
                        return True
                except Exception:
                    pass
                driver.switch_to.default_content()
            print("ℹ️  No iframe with payouts found; staying in top-level document")
            return False

        switch_to_payouts_frame()
        return True
        
    except Exception as e:
        print(f"❌ Error during login/navigation: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_payouts(driver, wait, csv_filename, existing_values):
    """Monitor payout values and write to CSV. Returns False if session is lost."""
    def switch_to_payouts_frame():
        driver.switch_to.default_content()
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        for idx, frame in enumerate(frames):
            try:
                driver.switch_to.frame(frame)
                if driver.find_elements(By.CSS_SELECTOR, ".payouts-wrapper, .payouts-block .payout"):
                    return True
            except Exception:
                pass
            driver.switch_to.default_content()
        return False

    last_value = None
    
    while True:
        try:
            # Check if we're still in a valid context
            if not driver.find_elements(By.CSS_SELECTOR, "body"):
                driver.switch_to.default_content()
                switch_to_payouts_frame()

            # Try multiple selectors for payout elements
            payout_elements = driver.find_elements(By.CSS_SELECTOR, ".payouts-wrapper .payouts-block .payout")
            if not payout_elements:
                payout_elements = driver.find_elements(By.CSS_SELECTOR, ".payouts-block .payout")
            if not payout_elements:
                payout_elements = driver.find_elements(By.CSS_SELECTOR, "app-stats-widget .payout")
            if not payout_elements:
                payout_elements = driver.find_elements(By.CSS_SELECTOR, ".payouts-wrapper .payout")
            if not payout_elements:
                payout_elements = driver.find_elements(By.CSS_SELECTOR, ".payout")

            if payout_elements:
                first_payout = payout_elements[0]
                if first_payout.is_displayed():
                    payout_text = first_payout.text.strip()
                    multiplier = payout_text.replace("x", "").replace(",", "").strip() if payout_text else ""

                    if multiplier and multiplier != last_value:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        value_key = (timestamp, multiplier)
                        if value_key not in existing_values:
                            with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([timestamp, multiplier])
                            existing_values.add(value_key)
                            last_value = multiplier
                            print(f"   ✅ Logged: {multiplier}x at {timestamp}")
                        else:
                            last_value = multiplier
            else:
                print("   ⏳ No payout elements found yet; retrying...")

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n⚠️  Monitoring stopped by user (Ctrl+C)")
            return None  # Signal to stop completely
        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"   ⚠️  WebDriver session error: {e}")
            print("   ❌ Session lost - will restart...")
            return False  # Signal to restart session
        except Exception as e:
            error_str = str(e).lower()
            if "session" in error_str or "invalid" in error_str:
                print(f"   ⚠️  Session-related error: {e}")
                print("   ❌ Session lost - will restart...")
                return False  # Signal to restart session
            print(f"   ⚠️  Monitoring error: {e}")
            time.sleep(2)

# Main execution with automatic recovery
def main():
    """Main function with automatic session recovery."""
    driver = None
    max_restart_attempts = 5
    restart_count = 0
    
    # Initialize CSV file
    existing_csv_files = glob.glob("aviator_payouts_*.csv")
    csv_filename = None
    existing_values = set()
    
    if existing_csv_files:
        csv_filename = max(existing_csv_files, key=os.path.getmtime)
        print(f"📂 Found existing CSV file: {csv_filename}")
        try:
            with open(csv_filename, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        existing_values.add((row[0], row[1]))
            print(f"✅ Loaded {len(existing_values)} existing records from CSV")
        except Exception as e:
            print(f"⚠️  Could not read existing CSV: {e}")
            existing_values = set()
    else:
        csv_filename = f"aviator_payouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "multiplier"])
        print(f"✅ New CSV file created: {csv_filename}")
    
    while restart_count < max_restart_attempts:
        try:
            # Clean up previous driver if exists
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                driver = None
            
            # Create new driver
            print(f"\n🚀 Starting new session (Attempt {restart_count + 1}/{max_restart_attempts})...")
            driver = create_driver()
            wait = WebDriverWait(driver, 20)
            
            # Login and navigate to Aviator
            if not login_and_navigate_to_aviator(driver, wait):
                print("❌ Failed to login/navigate. Retrying...")
                restart_count += 1
                time.sleep(5)
                continue
            
            print("📊 Starting continuous monitoring of the latest payout...")
            print("   Will log timestamp and multiplier to CSV until stopped (Ctrl+C).")
            
            # Start monitoring
            result = monitor_payouts(driver, wait, csv_filename, existing_values)
            
            if result is None:
                # User interrupted
                print(f"📁 CSV saved to {csv_filename}")
                break
            elif result is False:
                # Session lost - restart
                print(f"🔄 Session lost. Restarting in 5 seconds... (Attempt {restart_count + 1}/{max_restart_attempts})")
                restart_count += 1
                time.sleep(5)
                continue
                
        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user (Ctrl+C)")
            print(f"📁 CSV saved to {csv_filename}")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            restart_count += 1
            if restart_count < max_restart_attempts:
                print(f"🔄 Restarting in 5 seconds... (Attempt {restart_count + 1}/{max_restart_attempts})")
                time.sleep(5)
            else:
                print("❌ Max restart attempts reached. Exiting.")
                break
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

if __name__ == "__main__":
    main()
