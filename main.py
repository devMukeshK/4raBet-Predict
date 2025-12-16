from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import csv
import os
import glob
from datetime import datetime, timedelta
import pandas as pd

# --- Config ---
# Choose browser: "firefox", "chrome", or "safari"
BROWSER = "chrome"  # Recommended: Firefox is often faster for automation

LOGIN_URL = "https://4rabet365.com/"  # homepage with login button
USERNAME = "7979787871"
PASSWORD = "Adidas@1234"
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
global_csv_filename = os.path.join(data_folder, "aviator_payouts_global.csv")
current_csv_filename = os.path.join(data_folder, f"aviator_payouts_{datetime.now().strftime('%Y%m%d')}.csv")
# Initialize browser based on selection
driver = None
if BROWSER.lower() == "firefox":
    options = FirefoxOptions()
    # options.add_argument('--headless')  # Uncomment to run headless
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference("useAutomationExtension", False)
    driver = webdriver.Firefox(options=options)
    print("🦊 Using Firefox browser")
elif BROWSER.lower() == "safari":
    options = SafariOptions()
    # Safari doesn't support headless mode
    driver = webdriver.Safari(options=options)
    print("🦁 Using Safari browser")
else:  # Default to Chrome
    options = ChromeOptions()
    # options.add_argument('--headless')  # Uncomment to run headless
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    print("🌐 Using Chrome browser")

wait = WebDriverWait(driver, 20)

try:
    print("🌐 Step 1: Opening website...")
    driver.get(LOGIN_URL)
    time.sleep(2)  # Wait for page to load
    
    # Close any popup if present
    print("🔍 Step 2: Checking for popups...")
    try:
        close_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.close-btn")))
        close_button.click()
        print("✅ Popup closed")
        time.sleep(1)
    except TimeoutException:
        print("ℹ️  No popup found, continuing...")

    # 2. Click Log In button
    print("🔍 Step 3: Looking for Login button...")
    login_button = wait.until(EC.presence_of_element_located((By.ID, "auth_btn")))
    print("✅ Login button found")
    
    # Scroll into view
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", login_button)
    time.sleep(1)
    
    # Try to click - use JavaScript if regular click fails
    try:
        wait.until(EC.element_to_be_clickable(login_button))
        login_button.click()
        print("✅ Login button clicked (regular click)")
    except Exception as e:
        print(f"   Regular click failed ({type(e).__name__}), using JavaScript click...")
        driver.execute_script("arguments[0].click();", login_button)
        print("✅ Login button clicked (JavaScript click)")
    
    time.sleep(3)

    # 3. Try to click "Provide" button (optional step) - use shorter timeout
    print("🔍 Step 4: Checking for 'Provide' button (optional)...")
    provide_btn = None
    short_wait = WebDriverWait(driver, 3)  # Shorter timeout for optional element
    try:
        provide_btn = short_wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Provide')]")))
        provide_btn.click()
        print("✅ Provide button clicked")
        time.sleep(2)
    except TimeoutException:
        print("ℹ️  Provide button not found - this may be normal, proceeding to login form...")
        time.sleep(1)

    # 4. Wait for modal/dialog to appear (optional check)
    print("🔍 Step 5: Waiting for login form to appear...")
    try:
        short_wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
        print("✅ Login modal visible")
    except TimeoutException:
        print("ℹ️  Modal selector not found, proceeding to look for input fields directly...")
    
    time.sleep(2)  # Give time for form to fully load

    # Wait for phone input to be visible and stable
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
            raise Exception("Could not find phone input field. Please check the screenshot.")

    # Scroll into view
    driver.execute_script("arguments[0].scrollIntoView(true);", phone_input)
    time.sleep(0.5)

    # Set value via JS to ensure Vue sees it
    print("📝 Step 7: Entering phone number...")
    driver.execute_script("""
    let input = arguments[0];
    input.value = arguments[1];
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    """, phone_input, USERNAME)
    print(f"✅ Phone number entered: {USERNAME}")
    time.sleep(2)  # Wait for password field to appear after phone entry
    
    print("🔍 Step 8: Looking for password input field...")
    password_input = None
    try:
        # Try the specific data attribute first
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
                # Try finding by the label "Password" nearby
                password_input = wait.until(EC.presence_of_element_located((By.XPATH, "//label[contains(text(), 'Password')]/following::input[@type='password'] | //input[@type='password' and @data-auto-test-el='enterPassword']")))
                print("✅ Password input found (by label)")
            except TimeoutException:
                raise Exception("Could not find password input field. Please check the screenshot.")
    
    # Ensure it's visible
    if not password_input.is_displayed():
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", password_input)
        time.sleep(0.5)
    print("✅ Password field is ready")
    
    # Scroll password field into view and focus it
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", password_input)
    time.sleep(0.5)
    
    # Click and focus the password field first
    try:
        password_input.click()
        password_input.clear()
        print("✅ Password field focused and cleared")
    except Exception as e:
        print(f"   Note: Could not click password field directly ({type(e).__name__}), using JavaScript focus...")
        driver.execute_script("arguments[0].focus();", password_input)
        driver.execute_script("arguments[0].value = '';", password_input)
    
    time.sleep(0.5)
    
    print("📝 Step 9: Entering password...")
    # Try multiple methods to ensure password is entered
    try:
        # Method 1: Direct input via Selenium
        password_input.send_keys(PASSWORD)
        print("✅ Password entered via send_keys")
    except Exception as e:
        print(f"   send_keys failed ({type(e).__name__}), trying JavaScript...")
        # Method 2: JavaScript with all events
        driver.execute_script("""
        let input = arguments[0];
        input.focus();
        input.value = arguments[1];
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.dispatchEvent(new Event('blur', { bubbles: true }));
        """, password_input, PASSWORD)
        print("✅ Password entered via JavaScript")
    
    # Verify password was entered
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

    # Click Login
    print("🔍 Step 10: Looking for Sign In button...")
    
    # Wait for button to be present and clickable - try multiple selectors (prioritize login-sign-in-btn ID)
    login_btn = None
    try:
        # First try the specific login button ID
        login_btn = wait.until(EC.presence_of_element_located((By.ID, "login-sign-in-btn")))
        print("✅ Sign In button found (login-sign-in-btn ID)")
    except TimeoutException:
        try:
            # Try data-auto-test-el attribute
            login_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-auto-test-el="signIn"]')))
            print("✅ Sign In button found (data-auto-test-el)")
        except TimeoutException:
            try:
                print("   Trying button by wrapper...")
                # Try finding via wrapper
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
    
    # Wait for button to be enabled (it might be disabled until form is valid)
    if not login_btn.is_enabled():
        print("⚠️  Button is disabled, waiting for it to become enabled...")
        try:
            wait.until(lambda d: login_btn.is_enabled())
            print("✅ Button is now enabled")
        except TimeoutException:
            print("⚠️  Button still disabled after wait - checking form validation...")
            # Check if there are validation errors
            try:
                error_msgs = driver.find_elements(By.CSS_SELECTOR, ".v-messages__message, .error-message")
                for err in error_msgs:
                    if err.is_displayed():
                        print(f"   Validation error: {err.text}")
            except:
                pass
    
    # Verify form fields are filled before clicking
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
    time.sleep(1)  # Give form time to validate
    
    # Wait for button to be clickable
    try:
        wait.until(EC.element_to_be_clickable(login_btn))
        print("✅ Sign In button is clickable")
    except TimeoutException:
        print("⚠️  Button not clickable via standard check, will try JavaScript click...")
    
    # Scroll button into view - scroll the wrapper div instead
    try:
        wrapper = driver.find_element(By.CSS_SELECTOR, '[data-auto-test-el="auth-dialog-submit-wrapper"]')
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", wrapper)
    except:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", login_btn)
    time.sleep(1)
    
    # Remove any overlays that might be blocking (both modal overlay and button overlay)
    try:
        overlay = driver.find_element(By.CSS_SELECTOR, ".v-overlay__scrim")
        if overlay.is_displayed():
            print("   Removing overlay scrim...")
            driver.execute_script("arguments[0].style.display = 'none';", overlay)
            time.sleep(0.5)
    except:
        pass
    
    # Remove button's internal overlay if present
    try:
        btn_overlay = login_btn.find_element(By.CSS_SELECTOR, ".v-btn__overlay")
        if btn_overlay.is_displayed():
            print("   Removing button overlay...")
            driver.execute_script("arguments[0].style.pointerEvents = 'none';", btn_overlay)
    except:
        pass
    
    # Try multiple click strategies
    clicked = False
    
    # Strategy 1: Click on the button's content span (bypasses overlay)
    try:
        print("   Attempting click on button content span...")
        content_span = login_btn.find_element(By.CSS_SELECTOR, ".v-btn__content")
        content_span.click()
        clicked = True
        print("✅ Sign in button clicked successfully (via content span)!")
    except Exception as e:
        print(f"   Content span click failed ({type(e).__name__})")
    
    # Strategy 2: Regular Selenium click
    if not clicked:
        try:
            print("   Attempting regular click...")
            login_btn.click()
            clicked = True
            print("✅ Sign in button clicked successfully (regular click)!")
        except Exception as e:
            print(f"   Regular click failed ({type(e).__name__})")
    
    # Strategy 3: JavaScript click with proper event handling
    if not clicked:
        try:
            print("   Attempting JavaScript click with events...")
            driver.execute_script("""
            let btn = arguments[0];
            // Remove pointer-events from overlays
            let overlays = btn.querySelectorAll('.v-btn__overlay, .v-btn__underlay');
            overlays.forEach(ov => ov.style.pointerEvents = 'none');
            // Focus and click
            btn.focus();
            btn.click();
            """, login_btn)
            clicked = True
            print("✅ Sign in button clicked successfully (JavaScript click)!")
        except Exception as e2:
            print(f"   JavaScript click failed ({type(e2).__name__})")
    
    # Strategy 4: Dispatch click event directly with MouseEvent
    if not clicked:
        try:
            print("   Attempting direct MouseEvent dispatch...")
            driver.execute_script("""
            let btn = arguments[0];
            btn.focus();
            // Create and dispatch mouse events
            let mouseDown = new MouseEvent('mousedown', { bubbles: true, cancelable: true, view: window });
            let mouseUp = new MouseEvent('mouseup', { bubbles: true, cancelable: true, view: window });
            let clickEvent = new MouseEvent('click', { bubbles: true, cancelable: true, view: window });
            btn.dispatchEvent(mouseDown);
            btn.dispatchEvent(mouseUp);
            btn.dispatchEvent(clickEvent);
            """, login_btn)
            clicked = True
            print("✅ Sign in button clicked successfully (MouseEvent dispatch)!")
        except Exception as e3:
            print(f"   MouseEvent dispatch failed ({type(e3).__name__})")
    
    if not clicked:
        print("❌ All click methods failed!")
        driver.save_screenshot("click_failed_debug.png")
        print("🖼️  Saved screenshot to click_failed_debug.png")
    else:
        print("✅ Sign in button click completed!")
        time.sleep(2)  # Wait for form submission to start
        
        # Check for error messages
        try:
            error_elements = driver.find_elements(By.CSS_SELECTOR, ".error, .v-messages__message, [role='alert'], .text-error")
            if error_elements:
                for error in error_elements:
                    if error.is_displayed() and error.text.strip():
                        print(f"⚠️  Error message found: {error.text}")
        except:
            pass
        
        # Wait for sign-in to process - check if button becomes disabled (loading state)
        try:
            WebDriverWait(driver, 2).until(lambda d: not login_btn.is_enabled())
            print("⏳ Sign-in button disabled - form is submitting...")
        except:
            pass
        
        # Wait longer for login to complete
        print("⏳ Waiting for login to complete...")
        time.sleep(5)
        
        # Verify login succeeded by checking if modal disappeared or page changed
        try:
            # Check if login modal is gone
            WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
            print("✅ Login modal closed - login appears successful!")
        except TimeoutException:
            print("⚠️  Login modal still visible after 10 seconds")
            # Check for any error messages again
            try:
                error_elements = driver.find_elements(By.CSS_SELECTOR, ".error, .v-messages__message, [role='alert'], .text-error, .v-field__details")
                found_errors = False
                for error in error_elements:
                    if error.is_displayed() and error.text.strip():
                        print(f"❌ Error: {error.text}")
                        found_errors = True
                if not found_errors:
                    print("   No error messages found - login may still be processing")
            except:
                pass
            
            # Check if we're still on the login page or if URL changed
            current_url = driver.current_url
            print(f"   Current URL: {current_url}")
            
            driver.save_screenshot("login_status_debug.png")
            print("🖼️  Saved screenshot to login_status_debug.png")
        
        time.sleep(2)  # Additional wait
    
    # Wait for dashboard to load after successful login
    print("🔍 Step 11: Waiting for dashboard to load...")
    dashboard_loaded = False
    
    # Check for dashboard indicators (modal closed, URL changed, or dashboard elements visible)
    try:
        # Wait for login modal to disappear (indicates successful login)
        WebDriverWait(driver, 15).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__content.auth-dialog")))
        print("✅ Login modal closed - dashboard should be loading...")
        dashboard_loaded = True
    except TimeoutException:
        print("⚠️  Login modal still visible, but proceeding to check dashboard...")
    
    # Additional checks for dashboard
    if not dashboard_loaded:
        try:
            # Check if URL changed (might redirect to dashboard)
            current_url = driver.current_url
            if "login" not in current_url.lower() and "auth" not in current_url.lower():
                print(f"✅ URL changed to: {current_url} - likely on dashboard")
                dashboard_loaded = True
        except:
            pass
    
    # Wait for dashboard elements to appear
    try:
        # Look for common dashboard elements
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body, main, .dashboard, .main-content, [class*='dashboard'], [class*='main']")))
        print("✅ Dashboard elements detected")
        dashboard_loaded = True
    except TimeoutException:
        print("⚠️  Dashboard elements not found, but proceeding...")
    
    if dashboard_loaded:
        print("✅ Dashboard loaded successfully!")
    else:
        print("⚠️  Dashboard status unclear, but proceeding with wait...")
    
    # Step 12: Click on Aviator button after login
    print("🔍 Step 12: Looking for Aviator button...")
    aviator_clicked = False

    # Make sure we’re at the top-level document
    driver.switch_to.default_content()

    def find_aviator():
        """Return a fresh clickable element for Aviator (anchor preferred)."""
        try:
            anchor = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'a[href="/casino/slot/aviator"], a[href*="aviator"]')
                )
            )
            return anchor
        except TimeoutException:
            div = wait.until(EC.presence_of_element_located((By.ID, "top-section__aviator")))
            try:
                return div.find_element(By.XPATH, "./ancestor::a[1]")
            except Exception:
                return div

    try:
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
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});",
                    target_clickable,
                )
                time.sleep(0.5)

                wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//*[@id='top-section__aviator'] | //a[contains(@href,'aviator')]",
                        )
                    )
                )
                target_clickable.click()
                aviator_clicked = True
                print("✅ Aviator clicked successfully (regular click)")
                break

            except StaleElementReferenceException:
                print(f"   ⚠️ Stale element on attempt {attempt + 1}, retrying with a fresh element...")
                continue

            except Exception as e:
                print(f"   Regular click failed ({type(e).__name__}), trying JavaScript click...")
                try:
                    target_clickable = find_aviator()
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});",
                        target_clickable,
                    )
                    time.sleep(0.2)
                    driver.execute_script("arguments[0].click();", target_clickable)
                    aviator_clicked = True
                    print("✅ Aviator clicked successfully (JavaScript click)")
                    break
                except StaleElementReferenceException:
                    print(f"   ⚠️ JS click stale on attempt {attempt + 1}, retrying...")
                    continue
                except Exception as e2:
                    print(f"   JavaScript click failed ({type(e2).__name__}) on attempt {attempt + 1}")
                    continue

        if aviator_clicked:
            time.sleep(60)  # Wait for Aviator page to fully load

            # Try to switch into an iframe that contains the payouts, if any
            def switch_to_payouts_frame():
                driver.switch_to.default_content()
                frames = driver.find_elements(By.TAG_NAME, "iframe")
                for idx, frame in enumerate(frames):
                    try:
                        driver.switch_to.frame(frame)
                        # probe for payouts wrapper
                        if driver.find_elements(By.CSS_SELECTOR, ".payouts-wrapper, .payouts-block .payout"):
                            print(f"✅ Switched to iframe {idx} containing payouts UI")
                            return True
                    except Exception:
                        pass
                    driver.switch_to.default_content()
                print("ℹ️  No iframe with payouts found; staying in top-level document")
                return False

            switch_to_payouts_frame()

            # --- Continuous monitoring of first payout value ---
            print("📊 Starting continuous monitoring of the latest payout (first item)...")
            print("   Will log timestamp and multiplier to CSV until stopped (Ctrl+C).")

            # Initialize dual CSV file system using pandas
            # 1. global_csv_filename - stores ALL multipliers (all-time cumulative)
            # 2. current_csv_filename - stores data from previous day (daily file)
            
            df_global = None
            df_current = None
            
            # Initialize global CSV (file 1) - stores all multipliers
            if not os.path.exists(global_csv_filename):
                print(f"📂 Global CSV doesn't exist. Creating: {global_csv_filename}")
                df_global = pd.DataFrame(columns=['timestamp', 'multiplier'])
                df_global.to_csv(global_csv_filename, index=False)
            else:
                print(f"📂 Found global CSV: {global_csv_filename}")
                try:
                    df_global = pd.read_csv(global_csv_filename)
                    df_global['timestamp'] = pd.to_datetime(df_global['timestamp'])
                    print(f"✅ Loaded {len(df_global)} existing records from global CSV")
                except Exception as e:
                    print(f"⚠️  Could not read global CSV: {e}")
                    df_global = pd.DataFrame(columns=['timestamp', 'multiplier'])
            
            # Initialize current CSV (file 2) - stores data from previous day
            if not os.path.exists(current_csv_filename):
                print(f"📂 Current CSV doesn't exist. Creating: {current_csv_filename}")
                df_current = pd.DataFrame(columns=['timestamp', 'multiplier'])
                
                # Copy data from global CSV from previous date to latest
                if df_global is not None and len(df_global) > 0:
                    try:
                        yesterday = datetime.now() - timedelta(days=1)
                        # Filter data from yesterday onwards
                        df_recent = df_global[df_global['timestamp'] >= yesterday].copy()
                        if len(df_recent) > 0:
                            df_current = df_recent.copy()
                            df_current.to_csv(current_csv_filename, index=False)
                            print(f"✅ Copied {len(df_recent)} records from global CSV (from previous date) to current CSV")
                        else:
                            df_current.to_csv(current_csv_filename, index=False)
                            print(f"✅ Created empty current CSV (no recent data in global CSV)")
                    except Exception as e:
                        print(f"⚠️  Could not copy data from global CSV: {e}")
                        df_current.to_csv(current_csv_filename, index=False)
                else:
                    df_current.to_csv(current_csv_filename, index=False)
            else:
                print(f"📂 Found current CSV: {current_csv_filename}")
                try:
                    df_current = pd.read_csv(current_csv_filename)
                    df_current['timestamp'] = pd.to_datetime(df_current['timestamp'])
                    print(f"✅ Loaded {len(df_current)} existing records from current CSV")
                except Exception as e:
                    print(f"⚠️  Could not read current CSV: {e}")
                    df_current = pd.DataFrame(columns=['timestamp', 'multiplier'])

            last_value = None

            while True:
                try:
                    # ensure we are in a context; if body missing, reset
                    if not driver.find_elements(By.CSS_SELECTOR, "body"):
                        driver.switch_to.default_content()
                        switch_to_payouts_frame()

                    # Try multiple selectors for the payout elements (newest is first)
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
                            multiplier = (
                                payout_text.replace("x", "").replace(",", "").strip()
                                if payout_text
                                else ""
                            )

                            if multiplier and multiplier != last_value:
                                timestamp = datetime.now()
                                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Create new row
                                new_row = pd.DataFrame({
                                    'timestamp': [timestamp_str],
                                    'multiplier': [float(multiplier)]
                                })
                                
                                # Check if this record already exists in current CSV
                                exists_current = False
                                if df_current is not None and len(df_current) > 0:
                                    exists_current = ((df_current['timestamp'].astype(str) == timestamp_str) & 
                                                     (df_current['multiplier'].astype(float) == float(multiplier))).any()
                                
                                # Write to current CSV first (file 2) - append if not exists
                                if not exists_current:
                                    try:
                                        if df_current is None or len(df_current) == 0:
                                            df_current = new_row.copy()
                                        else:
                                            df_current = pd.concat([df_current, new_row], ignore_index=True)
                                        # Use mode='a' with header=False for faster append (but we'll use to_csv for simplicity)
                                        df_current.to_csv(current_csv_filename, index=False)
                                        print(f"   ✅ Logged to current CSV: {multiplier}x at {timestamp_str}")
                                    except Exception as e:
                                        print(f"   ⚠️  Error writing to current CSV: {e}")
                                
                                # Check if this record already exists in global CSV
                                exists_global = False
                                if df_global is not None and len(df_global) > 0:
                                    exists_global = ((df_global['timestamp'].astype(str) == timestamp_str) & 
                                                    (df_global['multiplier'].astype(float) == float(multiplier))).any()
                                
                                # Then write to global CSV (file 1) - append if not exists
                                if not exists_global:
                                    try:
                                        if df_global is None or len(df_global) == 0:
                                            df_global = new_row.copy()
                                        else:
                                            df_global = pd.concat([df_global, new_row], ignore_index=True)
                                        df_global.to_csv(global_csv_filename, index=False)
                                        print(f"   ✅ Logged to global CSV: {multiplier}x at {timestamp_str}")
                                    except Exception as e:
                                        print(f"   ⚠️  Error writing to global CSV: {e}")
                                
                                last_value = multiplier
                    else:
                        print("   ⏳ No payout elements found yet; retrying...")

                    time.sleep(1)

                except KeyboardInterrupt:
                    print("\n⚠️  Monitoring stopped by user (Ctrl+C)")
                    print(f"📁 Global CSV: {global_csv_filename}")
                    print(f"📁 Current CSV: {current_csv_filename}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Monitoring error: {e}")
                    time.sleep(2)
        else:
            print("❌ Aviator element could not be clicked after retries.")
            driver.save_screenshot("aviator_click_failed.png")
            print("🖼️  Saved screenshot to aviator_click_failed.png")

    except TimeoutException:
        print("❌ Aviator element not found!")
        driver.save_screenshot("aviator_not_found.png")
        print("🖼️  Saved screenshot to aviator_not_found.png")

    print("🎉 Login completed; Aviator click attempted!")


except TimeoutException as e:
    print(f"❌ Timeout: Element not found in time. Error: {e}")
    driver.save_screenshot("timeout_debug.png")
    print("🖼️  Saved screenshot to timeout_debug.png")
    print("⏳ Keeping browser open for 10 seconds for inspection...")
    time.sleep(10)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    driver.save_screenshot("error_debug.png")
    print("🖼️  Saved screenshot to error_debug.png")
    print("⏳ Keeping browser open for 10 seconds for inspection...")
    time.sleep(10)

finally:
    print("🔒 Closing browser...")
    driver.quit()

