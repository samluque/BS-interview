.PHONY: report report_html slides slides_charts clean

report: report_html

report_html:
	mkdir -p reports
	jupyter nbconvert --to html --execute notebooks/01_ad_performance_case.ipynb --output-dir reports --output 01_ad_performance_case.html

# Export charts for slides
slides_charts:
	python slides/export_charts.py

# Build PDF slides (requires pdflatex and metropolis theme)
slides: slides_charts
	cd slides && pdflatex -interaction=nonstopmode presentation.tex
	cd slides && pdflatex -interaction=nonstopmode presentation.tex

clean:
	rm -rf reports/*.html
	rm -rf slides/figures/*.png
	rm -rf slides/*.aux slides/*.log slides/*.nav slides/*.out slides/*.snm slides/*.toc slides/*.pdf

