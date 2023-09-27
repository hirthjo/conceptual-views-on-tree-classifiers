(defproject tree2ctx "0.2.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [org.clojure/data.csv "1.0.1"]
		 [conexp-clj "2.3.0-smeasure"]] ; make sure to use the smeasure branch
  :main ^:skip-aot tree2ctx.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
