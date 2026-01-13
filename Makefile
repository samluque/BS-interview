.PHONY: report report_html clean

report: report_html

report_html:
	mkdir -p reports
	jupyter nbconvert --to html --execute notebooks/01_ad_performance_case.ipynb --output-dir reports --output 01_ad_performance_case.html

clean:
	rm -rf reports/*.html


