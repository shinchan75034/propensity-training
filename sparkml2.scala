//run this: /grid/datashare/kt6238/pgm/sparkling-water-2.1.14/bin/sparkling-shell --num-executors 100 --executor-memory 64g --driver-memory 64g --master yarn-client /grid/datashare/kt6238/pgm/sparkling-water-2.1.14/assembly/build/libs/*.jar --conf spark.driver.maxResultSize=3g --conf spark.yarn.executor.memoryOverhead=3g
// or
// spark-shell --num-executors 100 --executor-memory 64g --driver-memory 64g --master yarn-client --conf spark.driver.maxResultSize=3g --jars /grid/datashare/kt6238/pgm/sparkling-water-2.1.14/assembly/build/libs/*.jar --conf spark.yarn.executor.memoryOverhead=3g

import java.io.File

import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import DeepLearningParameters.Activation
import DeepLearningParameters.Loss
import org.apache.spark.SparkFiles
import org.apache.spark.h2o.{DoubleHolder, H2OContext, H2OFrame}
import org.apache.spark.sql.Dataset
import water.support.{H2OFrameSupport, SparkContextSupport, SparkSessionSupport}
import org.apache.spark.examples.h2o.AirlinesParse
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.rand

// Run H2O cluster inside Spark cluster
val h2oContext = H2OContext.getOrCreate(sc)
import h2oContext._
import h2oContext.implicits._

val dat = spark.sqlContext.sql("select * from u76.acx_gold_1_vs")
val dat = spark.sqlContext.sql("select mob_ban, cast(estimated_hh_income as int), cast(household_size as int), cast(number_adults as int), cast(number_children as int), exposurestatus from u76.acx_gold_1_vs")

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
val lr = new LogisticRegressionWithLBFGS
lr.setIntercept(true)

val sp = dat.randomSplit(Array(0.2, 0.2, 0.6))
val (train, test, hold) = (sp(0), sp(1), sp(2))

train.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/kt6238/db/u76/for_logistic_regression_train").saveAsTable("kt6.for_logistic_regression_train")
test.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/kt6238/db/u76/for_logistic_regression_test").saveAsTable("kt6.for_logistic_regression_test")
hold.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/kt6238/db/u76/for_logistic_regression_hold").saveAsTable("kt6.for_logistic_regression_hold")

val trainData = spark.sqlContext.sql("select mob_ban, cast(estimated_hh_income as int),cast(household_size as int),cast(number_adults as int),cast(number_children as int),tvonly,webonly,tvandweb,cast(exposurestatus as int) from kt6.for_logistic_regression_train")

// deal with categorical data
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, IDF, VectorAssembler, HashingTF, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
def indexStringColumns(df:DataFrame, cols:Array[String]):DataFrame = {
    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.feature.StringIndexerModel
    //variable newdf will be updated several times
    var newdf = df
    for(c <- cols) {
        val si = new StringIndexer().setInputCol(c).setOutputCol(c+"-num")
        val sm:StringIndexerModel = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    }
    newdf
}

val good_data = trainData.na.drop(how = "any").drop("tvonly").drop("webonly").drop("tvandweb")


val dfnumeric = indexStringColumns(good_data, Array("estimated_hh_income", "household_size", "number_adults", "number_children","exposurestatus"))

def oneHotEncodeColumns(df:DataFrame, cols:Array[String]):DataFrame = {
    import org.apache.spark.ml.feature.OneHotEncoder
    var newdf = df
    for(c <- cols) {
        val onehotenc = new OneHotEncoder().setInputCol(c)
        onehotenc.setOutputCol(c+"-onehot").setDropLast(false)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    }
    newdf
}


val df_hot = oneHotEncodeColumns(dfnumeric, Array("estimated_hh_income", "household_size", "number_adults", "number_children"))

// isolate features and target variables.
val va = new VectorAssembler().setOutputCol("features").setInputCols(df_hot.columns.diff(Array("exposurestatus", "mob_ban")))

val lpoints = va.transform(df_hot).select("features", "exposurestatus", "mob_ban").
      withColumnRenamed("exposurestatus", "label")

  // training the model
  val splits = lpoints.randomSplit(Array(0.8, 0.2))
  val adulttrain = splits(0).cache()
  val adultvalid = splits(1).cache()

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.ml.classification.LogisticRegression
  val lr = new LogisticRegression()
  lr.setIntercept(true)

val lrmodel = lr.fit(adulttrain)
/*
17/12/19 19:56:42 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
17/12/19 19:56:42 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
lrmodel: org.apache.spark.ml.classification.LogisticRegressionModel = logreg_574df62cf95d
*/

// evaluate model performance.
val validpredicts = lrmodel.transform(adultvalid)

validpredicts.show(false)
/*
+----------------------------------+-----+----------------------------------------------------------------+----------------------------------------+-------------------------------------+----------+
|features                          |label|mob_ban                                                         |rawPrediction                           |probability                          |prediction|
+----------------------------------+-----+----------------------------------------------------------------+----------------------------------------+-------------------------------------+----------+
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |001A95594C617216E1C52C4CF6C80B5DC54DAEF12CBF503ADE38028F8F45C1E5|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |00E3F8B8E3222763B18D76CF51E0BF896E63EEE61FD8E97E0734B751A2546684|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |0285D7EC74418D292EDEEA3B8D539E0FF896BF0760437964E0158EF25E2C7A41|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |0417A379CBA2374BA877B20B745FA1B6BE7884C43A0BA4F4DFAFC801A09D59A8|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |047AA98EB9D097095885DBB71C0C94783536759DC8E1E858EEC72739237BAE86|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
|(33,[0,9,18,25],[1.0,1.0,1.0,1.0])|0.0  |0500D327622F6AE0B32110309D00623C2543463E0B38095F2476D22AFCCA4B49|[0.6555784672956375,-0.6555784672956375]|[0.658266454177485,0.341733545822515]|0.0       |
*/

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
val bceval = new BinaryClassificationEvaluator()

bceval.evaluate(validpredicts)
/*
res19: Double = 0.5639045306607103
*/

bceval.getMetricName
/*
res20: String = areaUnderROC
*/

// get build a truth TABLE

val trainpredicts = lrmodel.transform(adulttrain)
val traineval = new BinaryClassificationEvaluator()
traineval.evaluate(trainpredicts)
trainpredicts.show(false)

val df1 = trainpredicts.select("mob_ban", "probability")

import org.apache.spark.ml.feature.VectorDisassembler

val disassembler = new VectorDisassembler().setInputCol("probability")
val df2 = disassembler.transform(df1)
val df3 = df2.drop("probability").drop("probability_0").distinct

val j1 = trainData.join(df3, trainData("mob_ban")===df3("mob_ban")).drop(df3("mob_ban"))
val j2 = j1.select("mob_ban", "estimated_hh_income","household_size","number_adults","number_children","probability_1")


j2.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/u76/db/train_ps").saveAsTable("u76.train_ps")
