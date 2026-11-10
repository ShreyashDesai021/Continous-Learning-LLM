"""Optional browser checks: pip install playwright; python -m playwright install chromium.

Run: python scripts/browser_smoke.py [--chromium /path/to/browser]
Uses an isolated temporary database and ephemeral local port, not your real workspace.
"""
import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from enterprise.api import create_app
import uvicorn


def main():
    os.environ['EVOLLM_ADMIN_PASSWORD']='EvoLLM-Admin-2026!'
    os.environ['EVOLLM_EMPLOYEE_PASSWORD']='EvoLLM-Employee-2026!'
    os.environ['EVOLLM_RETRIEVAL']='lexical'
    os.environ['EVOLLM_DEMO']='0'
    parser=argparse.ArgumentParser()
    parser.add_argument('--chromium')
    parser.add_argument('--screenshots')
    args=parser.parse_args()
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as root:
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(create_app(root),host='127.0.0.1',port=port,log_level='warning'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        deadline=time.monotonic()+10
        while not server.started and time.monotonic()<deadline:time.sleep(.05)
        assert server.started,'Local server failed to start'
        errors=[];checks=[]
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path=args.chromium,
                    args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'],headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1000})
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('console',lambda m:errors.append(m.text) if m.type=='error' and '401' not in m.text else None)
                page.goto(f'http://127.0.0.1:{port}')
                if args.screenshots:
                    Path(args.screenshots).mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(Path(args.screenshots)/'login.png'),full_page=True,animations="disabled")
                page.locator('#username').fill('admin')
                page.locator('#password').fill('EvoLLM-Admin-2026!')
                page.locator('#loginForm button[type=submit]').click()
                page.locator('#workspace').wait_for(state='visible');checks.append('login')
                page.locator('#emptyWorkspace').wait_for(state='visible');checks.append('empty workspace onboarding')
                page.locator('[data-view=benchmark]').click()
                page.once('dialog',lambda d:d.accept())
                page.locator('#loadDemo').click()
                page.wait_for_function("() => document.querySelector('#documentCount').textContent==='20'")
                page.locator('[data-view=chat]').click()
                page.locator('[data-agent=compare]').click()
                page.locator('#question').fill('What is the current weekly remote-work maximum?')
                page.locator('#askButton').click()
                page.locator('.answer-card').nth(2).wait_for()
                assert page.locator('.answer-card').count()==3
                assert 'four' in page.locator('.answer-card').nth(1).inner_text();checks.append('three-way comparison')
                if args.screenshots:page.screenshot(path=str(Path(args.screenshots)/'comparison.png'),full_page=True,animations="disabled")
                # Open a real citation, including its version-change history.
                page.locator('.answer-card').nth(1).locator('details').first.locator(':scope > summary').click()
                page.locator('.answer-card').nth(1).locator('[data-source]').first.click()
                page.locator('#sourceDialog').wait_for(state='visible')
                assert 'Version history' in page.locator('#sourceContent').inner_text()
                page.locator('[data-close=sourceDialog]').click();checks.append('citation and lineage')
                page.locator('[data-view=documents]').click()
                assert page.locator('#docRows tr').count()==20
                page.locator('#openUpload').click()
                page.locator('#file').set_input_files({'name':'travel.md','mimeType':'text/markdown','buffer':b'# Travel allowance\n\nThe approved travel allowance is INR 900 per day.'})
                for key,value in {'title':'Travel policy','family':'travel','document_key':'travel','version':'1'}.items():
                    page.locator('#uploadForm [name='+key+']').fill(value)
                page.locator('#uploadSubmit').click()
                page.locator('#uploadDialog').wait_for(state='hidden')
                page.wait_for_function("() => document.querySelectorAll('#docRows tr').length===21")
                checks.append('file upload and document refresh')
                page.locator('[data-view=chat]').click()
                page.locator('[data-agent=amo]').click()
                page.locator('#question').fill('What is the travel allowance per day?')
                page.locator('#askButton').click()
                page.wait_for_function("() => document.querySelectorAll('.answer-card').length===4")
                assert '900' in page.locator('.answer-card').last.inner_text()
                checks.append('query freshly uploaded document')
                page.locator('#mobileSettings').click()
                page.locator('#openTeam').click()
                for key,value in {'username':'compliance_user','password':'Compliance-2026!','department':'Compliance'}.items():
                    page.locator('#teamForm [name='+key+']').fill(value)
                page.locator('#teamForm button[type=submit]').click()
                page.wait_for_function("() => document.querySelector('#teamRows').textContent.includes('compliance_user')")
                page.locator('[data-close=teamDialog]').click()
                page.locator('[data-close=settingsDialog]').click()
                checks.append('custom department account')
                page.locator('.answer-card').last.locator('[data-source]').first.click()
                page.locator('#editMetadata').click()
                page.locator('#metadataForm [name=status]').select_option('draft')
                page.locator('#metadataForm button[type=submit]').click()
                page.locator('#metadataDialog').wait_for(state='hidden')
                checks.append('approval metadata edit')

                page.locator('[data-view=memory]').click();page.locator('#memoryRows tbody tr').first.wait_for();checks.append('memory tiers')
                page.locator('[data-view=benchmark]').click();page.locator('#runBenchmark').click()
                page.locator('#benchmarkResults tbody tr').nth(3).wait_for();checks.append('25-question benchmark')
                page.locator('[data-view=history]').click();page.locator('[data-conversation]').first.wait_for();checks.append('saved history')
                page.locator('[data-view=audit]').click();page.locator('#auditRows tbody tr').first.wait_for();checks.append('audit log')
                page.set_viewport_size({'width':390,'height':844});page.locator('[data-view=chat]').click()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile overflow'
                if args.screenshots:page.screenshot(path=str(Path(args.screenshots)/'mobile.png'),full_page=True,animations="disabled")
                checks.append('mobile layout')
                assert not errors,errors
                browser.close()
            print(json.dumps({'passed':checks,'browser_errors':errors},indent=2))
        finally:
            server.should_exit=True;thread.join(timeout=5)

if __name__=='__main__':main()
