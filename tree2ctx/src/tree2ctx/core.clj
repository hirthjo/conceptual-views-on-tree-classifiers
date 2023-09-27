(ns tree2ctx.core
  (:require [conexp.fca.contexts :refer :all]
            [conexp.fca.smeasure :refer :all]
            [conexp.gui.draw :refer :all]
            [conexp.fca.lattices :refer :all]
            [conexp.base :refer :all]
            [conexp.io.contexts :refer :all]
            [conexp.io.layouts :refer :all]
            [conexp.io.many-valued-contexts :refer :all]
            [conexp.fca.many-valued-contexts :refer :all]
            [clojure.data.json :as json]
            [clojure.tools.cli :refer [parse-opts]]
            [conexp.layouts.base :refer [update-valuations]]
            [conexp.layouts.freese :refer :all]
            [conexp.fca.fast :as fast])
  (:gen-class))


(defn get-splits
  "Returns a map of all thresholds per feature name."
  [dleafs]
  (apply merge-with union
         (map #(reduce (fn [dict leaf]
                         (merge-with union 
                                     dict 
                                     {(get leaf "feature") #{(get leaf "threshold")}}))
                       {} (butlast %))
              dleafs)))

(defn get-scale
  "Returns the scale to be used for a many-valued attribute according to
  the trees splits. (biordinal by threshold values)"
  [mv-ctx splits attr]
  (make-context (values-of-attribute mv-ctx (symbol attr)) 
                (let [thresholds (get splits (str attr) #{})]
                  (if (empty? thresholds) #{#{}}
                      (apply union (for [t thresholds] #{['<= t] ['> t]}))))
                (fn [a b] (if (empty? b) true
                              ((resolve (first b)) (second b) a)))))

(defn scale-by-dleafs ;; interordinal predicate scale
  "Scales the many valued context according to the decision leafs
  splits (interordinal)."
  [mv-ctx dleafs]
  (let [splits (get-splits dleafs)]
    (scale-mv-context mv-ctx
                      (reduce
                       #(assoc %1 %2 (get-scale mv-ctx splits %2))
                       {} (attributes mv-ctx)))))

(defn treectx
  "Computes a context of decision leafs (objects) and their
  splits (attributes). The split attributes are the same as in the
  scaled-ctx, since they were used for scaling."
  [scaled-attributes dleafs]
  (let [dleafs-v (vec dleafs)]
    (make-context (-> dleafs count range) scaled-attributes
                  (fn [l [f t]] (and (not (empty? t))
                                     (some #(and (= (get % "feature") (str f))
                                                 (= (get % "threshold") (second t))
                                                 (or (and (not (get % "t<v")) (not (= (first t) '<=)))
                                                     (and (get % "t<v") (= (first t) '<=)))) (butlast (nth dleafs-v l))))))))

(defn node-decoder
  [node]
  [(symbol (get node "feature")) [(if (get node "t<v") '<= '>) (get node "threshold")]]) ; todo reverse

(defn leaf-to-nodes
  [leaf]
  (doall (reduce #(conj %1 (map node-decoder 
                                (take (inc %2) (butlast leaf)))) 
                 #{}
                 (range (count (butlast leaf))))))

(defn leafs-to-nodes
  [dleafs]
  (reduce #(into %1 (leaf-to-nodes %2)) #{} dleafs))

(defn tree-order-ctx
  "Computes a context of decision leafs (objects) and their
  splits (attributes). The split attributes are the same as in the
  scaled-ctx, since they were used for scaling."
  [scaled-ctx dleafs]
  (let [nodes (map vec (leafs-to-nodes dleafs))]
    (rename-attributes 
     (make-context (objects scaled-ctx) nodes
                   (fn [a predicates] 
                     (every? #(or ;((incidence scaled-ctx) [a (str %)])
                               ((incidence scaled-ctx) [a %]))
                             predicates)))
     str)))                                ;    #(str (.indexOf nodes %) (last %))
    

;todo disj class attribtues once they are added
(defn scale-measure-map 
  "Returns the map of the scale-measure. Maps each object to the
  decision leaf used for classification. "
  [ctx tree-ctx]
  (fn [o] 
    (first ; each object is only a model of a single leaf  
     (filter #(subset? (object-derivation tree-ctx #{%}) (object-derivation ctx #{o}))
             (objects tree-ctx))))) ;; very slow

(defn product-scale  ; tree predicate scale
  "Computes the product scale of the tree scaling."
  [tree-sm nom-scale]
  (let [G (-> tree-sm context objects)
        M (-> tree-sm scale attributes)
        I #(let [leaf (->> #{%1} (object-derivation nom-scale) first)] ;; nominal scales have derivation size 1
             ((-> tree-sm scale incidence) [leaf %2]))]
    (make-context G M I)))

(defn nominal-leaf-scale 
  "compute nominal scale induced by the decision leafs."
  [tree-sm]
  (let [G (-> tree-sm context objects)
        M (-> tree-sm scale objects)
        leaf-parts (reduce #(assoc %1 %2 (->> #{%2}
                                              (object-derivation (scale tree-sm))
                                              (attribute-derivation (context tree-sm)))) 
                           {}
                           (-> tree-sm scale objects))
        I #(contains? (get leaf-parts %2) %1)]
    (make-context G M I)))

(defn dumpClass 
  [mvctx dataName tree_num]
  (let [class-mv-ctx (make-mv-subcontext mvctx 
                       (objects mvctx) 
                       (intersection (attributes mvctx)
                                     #{'binaryClass 'target 'class}))] ; some have binaryClass
    (write-context :named-binary-csv 
                   (rename-attributes
                    (rename-objects 
                     (make-context (objects class-mv-ctx) 
                                   (values-of-attribute class-mv-ctx
                                                        (first (attributes class-mv-ctx)))
                                   (fn [a b] 
                                     (= b
                                        ((comp second first) 
                                         (incidences-of-object class-mv-ctx a)))))
                     str)
                    str)      
                   (str "../output/" dataName "/" tree_num "/" dataName "_class.csv"))))

(defn read-rf-scales
  [dataName tree_num]
  (let [path (str "../output/" dataName "/" tree_num "/RF/")
        dtree-scale (read-context (str path dataName "_dtree-order-scale.csv") 
                                  :named-binary-csv)
        ;; tree-scale (read-context (str path dataName "_tree-scale.csv") 
        ;;                          :named-binary-csv)
        interord-scale (read-context (str path dataName "_interordinal-scale.csv") 
                                     :named-binary-csv)
        product-scale (read-context (str path dataName "_product-scale.csv")
                                    :named-binary-csv)
        nominal-scale (read-context (str path dataName "_nominal-scale.csv") 
                                    :named-binary-csv)]
    {:dtree-scale dtree-scale
    ; :tree-scale tree-scale
     :interordinal-predicate-scale interord-scale
     :tree-predicate-scale product-scale
     :nominal-scale nominal-scale}))

(defn read-rf-reduced-scales
  [dataName tree_num]
  (let [path (str "../output/" dataName "/" tree_num "/RF/")
        dtree-scale-reduced (read-context (str path dataName "-reduced-_dtree-order-scale.csv") 
                                  :named-binary-csv)
        ;; tree-scale (read-context (str path dataName "_tree-scale.csv") 
        ;;                          :named-binary-csv)
        interord-scale-reduced (read-context (str path dataName "-reduced-_interordinal-scale.csv") 
                                     :named-binary-csv)
        product-scale-reduced (read-context (str path dataName "-reduced-_product-scale.csv")
                                    :named-binary-csv)
        nominal-scale-reduced (read-context (str path dataName "-reduced-_nominal-scale.csv") 
                                    :named-binary-csv)]
    {:dtree-scale dtree-scale-reduced
    ; :tree-scale tree-scale
     :interordinal-predicate-scale interord-scale-reduced
     :tree-predicate-scale product-scale-reduced
     :nominal-scale nominal-scale-reduced}))

(defn make-rf-scale
  [dataName tree_num]
  (let [path (str "../output/" dataName "/" tree_num "/")
        union-context (fn [scale-seq]
                        (make-context (-> scale-seq first objects)
                                      (apply union (map attributes scale-seq))
                                      (apply union (map incidence-relation scale-seq))))
        apposition-context (fn [scale-seq]
                             (let [inc-seq (map incidence scale-seq)]
                               (make-context (-> scale-seq first objects)
                                             (apply disjoint-union (map attributes scale-seq))
                                             (fn [o [m i]] ((nth inc-seq i) [o m])))))]
    (println "scale,obj,obj-reduced,attr,attr-red,inc,inc-reduced")
    (doseq [scale #{"_dtree-order-scale.csv" 
                    ;"_tree-scale.csv"   
                    "_interordinal-scale.csv"
                    "_product-scale.csv" 
                    "_nominal-scale.csv"}]
      (let [scale-ctx (if (= scale "_nominal-scale.csv")
                        (apposition-context
                         (map 
                          #(read-context (str path % "/" dataName scale) :named-binary-csv) 
                          (range tree_num)))
                        (union-context 
                         (map 
                          #(read-context (str path % "/" dataName scale) :named-binary-csv) 
                          (range tree_num))))
            reduced-scale-ctx (reduce-attributes scale-ctx)]        
        (do
          (println scale 
                   (count (objects scale-ctx))            (count (objects reduced-scale-ctx))
                   (count (attributes scale-ctx))         (count (attributes reduced-scale-ctx))
                   (count (incidence-relation scale-ctx)) (count (incidence-relation reduced-scale-ctx)) )
          (write-context :named-binary-csv
                           scale-ctx
                           (str path "RF/" dataName scale))
          (write-context :named-binary-csv
                           reduced-scale-ctx
                           (str path "RF/" dataName "-reduced-" scale)))))))

(defn compute_scales_for_idx
  [mv-data dataName tree_num tree_idx]
  (let [path (str "../output/" dataName "/" tree_num "/" tree_idx "/") 
        dleafs (-> (str path dataName "_dleafs.json") 
                      slurp
                      json/read-str)
          ;;; compute multiple scales ctx_nom <= product_scale <= ctx_train 
                                        ; scale all the many-valued context by the predicates given in the decision tree
        ctx (scale-by-dleafs mv-data dleafs)
        dtree-ctx (tree-order-ctx ctx dleafs) ; tree oder + bot
                                        ; dleafs to their annotated predicates
        tree-ctx (treectx (attributes ctx) dleafs) ; leafs to predicates
        sm (make-smeasure-nc ctx tree-ctx (scale-measure-map ctx tree-ctx))

                                        ; compute nominal scale induced by the decision leafs
        nominal-scale-ctx (nominal-leaf-scale sm)
;        o (println "nom" (count (incidence-relation nominal-scale-ctx)))
                                        ; scale where every object is only in incidence with a
                                        ; predicate if the preducate is used by the dleaf that
                                        ; classifies the object
        product-scale-ctx (product-scale sm nominal-scale-ctx)]
    (write-context :named-binary-csv 
                   dtree-ctx
                   (str path dataName "_dtree-order-scale.csv"))
    ;; (write-context :named-binary-csv
    ;;                tree-ctx 
    ;;                (str path dataName "_tree-scale.csv"))
    (write-context :named-binary-csv
                   ctx 
                   (str path dataName "_interordinal-scale.csv"))
    (write-context :named-binary-csv
                   product-scale-ctx
                   (str path dataName "_product-scale.csv"))
    (write-context :named-binary-csv
                   nominal-scale-ctx
                   (str path dataName "_nominal-scale.csv"))
    (println "Done Scaling " tree_idx)))


(defn compute_scales
  [mv_ctx dataName tree_num]
  (if (= 1 tree_num)
    (compute_scales_for_idx mv_ctx dataName 0)
    (loop [idx (dec tree_num)]
      (println idx)
      (compute_scales_for_idx mv_ctx dataName tree_num idx)
      (if (not= idx 0)
        (recur (dec idx))))))

(defn scale-error-analysis
  [dataName tree_num]
  (let [path (str "../output/" dataName "/"tree_num)
        mv-data_train (read-mv-context (str path "/" dataName "_train_enc.csv") :data-table)
        mv-data_test (read-mv-context (str path dataName "_test_enc.csv") :data-table)
          dleafs (asd-> dataName $
                        (str path "/" $ "_dleafs.json") 
                        slurp
                        json/read-str)
        ;;; compute multiple scales ctx_nom <= product_scale <= ctx_train 
          
                                        ; scale all the many-valued context by the predicates given in the decision tree
          ctx_train (scale-by-dleafs mv-data_train dleafs)
          ctx_test (scale-by-dleafs mv-data_test dleafs)
                                        ; dleafs to their annotated predicates
          tree-ctx (treectx (attributes ctx_train) dleafs)
          sm_train (make-smeasure-nc ctx_train tree-ctx (scale-measure-map ctx_train tree-ctx))
          sm_test (make-smeasure-nc ctx_test tree-ctx (scale-measure-map ctx_test tree-ctx))

                                        ; compute nominal scale induced by the decision leafs
          nominal-scale-train (nominal-leaf-scale sm_train)
          nominal-scale-test (nominal-leaf-scale sm_test)

                                        ; scale where every object is only in incidence with a
                                        ; predicate if the preducate is used by the dleaf that
                                        ; classifies the object
          product-scale-train (product-scale sm_train nominal-scale-train)
          product-scale-test (product-scale sm_test nominal-scale-test)

                                        ; compute scaling error
          error_train (error-in-smeasure sm_train)
          error_test (error-in-smeasure sm_test)]
        (dumpClass mv-data_test (str dataName "_test") tree_num)
        (dumpClass mv-data_train (str dataName "_train") tree_num)
        ;;return error
        (println (-> tree-ctx extents count) (count error_train) (count error_test))))

(defn- rf-bv-sizes
  [dataName tree_num] 
  (let [scales (read-rf-reduced-scales dataName tree_num)]
    (println "scale,concepts")
    (doseq [s (keys scales)]
      (let [c (->> s scales
           (fast/concepts :pcbo)
           count)]
      (println s "," c)))))

(defn -main
  "Reads in the data set and the decision leafs and computes the scale-measure error.
  Scaling error is computed for data set scaled by all predicates
  compared to the tree scale. The data set must not include missing values."
  [dataName & args]
  (let [{:keys [mode rf]} (:options 
                        (parse-opts args [["-m" "--mode MODE" "What code should be executed." :default "help"]
                                          ["-r" "--rf RF" "Number of Trees." :default 1 :parse-fn #(Integer/parseInt %)]]))]
    (case mode
      "help" (print (:doc (meta #'-main)))
      "CV" ; cross validation step
      (scale-error-analysis dataName rf) ;; not used currently
      "C" ; complete model
      (let [mv-data (read-mv-context (str "../output/" dataName "/" rf "/" dataName "_enc.csv") :data-table)]
        (dumpClass mv-data dataName rf)
        (compute_scales mv-data dataName rf))
      "RF"
      (make-rf-scale dataName rf)
      "CC"
      (rf-bv-sizes dataName rf) )))

;;;;; drawing functions


(defn draw-with-class-and-tree 
  [ctx class-ctx dtree-ctx]
  (let [Exts (extents dtree-ctx)]
    (draw-layout
     (update-valuations
      (freese-layout 
       (concept-lattice ctx))
      (fn [[A B]]
        (str (if (contains? Exts A) "DT " "")
           (frequencies
            (for [o A]
              ((comp second first #(disj % nil)) 
               (values-of-object class-ctx o))))))))))


(defn draw-local-scaling 
  [ctx g class-ctx dtree-ctx]
  (let [Exts (set(extents dtree-ctx))]
    (draw-layout
     (update-valuations
      (freese-layout 
       (make-lattice-nc 
        (filter #(contains? (first %) g)
                (object-concept-lattice-filter-and-covering ctx g))
        (fn [[a b] [a2 b2]] 
          (subset? a a2))))
      (fn [[A B]]
        (str (if (contains? Exts A) "DT " "")
             (frequencies
              (for [o A]
                ((comp second first #(disj % nil)) 
                 (values-of-object class-ctx o))))))))))

(defn draw-local-scaling+covering
  [ctx g class-ctx dtree-ctx]
  (let [Exts (extents dtree-ctx)]
    (draw-layout
     (update-valuations
      (freese-layout 
       (make-lattice-nc 
        (object-concept-lattice-filter-and-covering ctx g)
        (fn [[a b] [a2 b2]] 
          (subset? a a2))))
      (fn [[A B]]
        (str (if (contains? Exts A) "DT " "")
             (frequencies
              (for [o A]
                ((comp second first #(disj % nil)) 
                 (values-of-object class-ctx o))))))))))

(defn draw-local-scaling-to-file
  [ctx g class-ctx dtree-ctx file]
  (write-layout :tikz
                (let [Exts (extents dtree-ctx)]
                  (update-valuations
                   (freese-layout 
                    (make-lattice-nc 
                     (object-concept-lattice-filter-and-covering ctx g)
                     (fn [[a b] [a2 b2]] 
                       (subset? a a2))))
                   (fn [[A B]]
                     (str (if (contains? Exts A) "DT " "")
                          (frequencies
                           (for [o A]
                             ((comp second first #(disj % nil)) 
                              (values-of-object class-ctx o))))))))
                file))


;; (def dataName "bin-car.csv")
;; (def mv-data (read-mv-context (str "../output/" dataName "_enc.csv") :data-table))
;; (def  dleafs (asd-> dataName $
;;                                (str "../output/" $ "_dleafs.json") 
;;                                slurp
;;                                json/read-str))
;; (def ctx (scale-by-dleafs mv-data dleafs))
;; (def dtree-ctx (tree-order-ctx ctx dleafs)) ; tree oder + bot
;; (def tree-ctx (treectx (attributes ctx) dleafs)) ; leafs to predicates
;; (def sm (make-smeasure-nc ctx tree-ctx (scale-measure-map ctx tree-ctx)))
;; (def nominal-scale-ctx (nominal-leaf-scale sm))
;; (def product-scale-ctx (product-scale sm nominal-scale-ctx))
;; (def class-ctx (rename-objects 
;;                 (read-context (str "../output/" dataName "_class.csv") :named-binary-csv)
;;                 #(Integer/parseInt %)))
;; (draw-with-class-and-tree product-scale-ctx 428 class-ctx dtree-ctx)
;; check orientation of < and <=
