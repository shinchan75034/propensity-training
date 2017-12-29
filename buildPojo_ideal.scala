// run this
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

/*
val fileName = "/apps/advertising/audience_measurement/kt6238/u76/temp/for_logistic_regression_train_ps.txt"
val loc = "hdfs:///".concat(fileName)
val datRdd = sc.textFile(loc).map(line => line.split("\\|"))
// datRdd.filter(f => f(0) == null | f(1) == null | f(2) == null | f(3) == null).count // check data quality
// datRdd.filter(f => f.length != 4).count / /check for incorrect delimiter.
case class Dat(mob_ban: String = "", estimated_hh_income: Int = 0, household_size: Int = 0, number_adults: Int = 0, number_children: Int = 0, tvonly: String = "", webonly: String = "",
tvandweb: String = "", exposurestatus: Int = 0, predict: Int = 0, p0: Double = 0.0, p1: Double = 0.0)
val datDF = datRdd.map(f => Dat(f(0).toString, f(1).toInt, f(2).toInt, f(3).toInt, f(4).toInt, f(5).toString, f(6).toString, f(7).toString, f(8).toInt, f(9).toInt, f(10).toDouble, f(11).toDouble)).
toDF().drop("mob_ban", "tvonly", "webonly", "tvandweb", "exposurestatus", "predict", "p0")


datDF.createOrReplaceTempView("datTbl")

val good_data = datDF.na.drop(how = "any")
*/

val good_data = spark.sqlContext.sql("select * from u76.train_ps")

// try h2o
// split into three sets.
val newDF = good_data.randomSplit(Array(0.6, 0.2, 0.2))
val (train, test, validation) = (newDF(0), newDF(1), newDF(2))
// make H2OFrame
val train_h2o : H2OFrame = h2oContext.asH2OFrame(train)
val test_h2o  : H2OFrame = h2oContext.asH2OFrame(test)
val valid_h2o : H2OFrame = h2oContext.asH2OFrame(validation)

import _root_.hex.tree.drf.DRFModel.DRFParameters
import _root_.hex.tree.drf.DRFModel
import _root_.hex.tree.drf.DRF
val drfParams = new DRFParameters()
drfParams._train = train_h2o
drfParams._response_column = 'probability_1
drfParams._ntrees = 500
drfParams._balance_classes = true
drfParams._max_after_balance_size = 2.0f

val rf = new DRF(drfParams)
val model = rf.trainModel().get()

/*
model: hex.tree.drf.DRFModel =
Model Metrics Type: Regression
 Description: Metrics reported on Out-Of-Bag training samples
 model id: DRF_model_1513272247170_1
 frame id: frame_rdd_14_8760b7e5cf5e2c0d21fe021ffbbd4224
 MSE: 1.0981661E-4
 RMSE: 0.010479342
 mean residual deviance: 1.0981661E-4
 mean absolute error: 0.007898416
 root mean squared log error: 0.006516975
Variable Importances:
           Variable Relative Importance Scaled Importance Percentage
estimated_hh_income        35618.644531          1.000000   0.423846
     household_size        20426.888672          0.573489   0.243071
    number_children        15834.613281          0.444560   0.188425
      number_adults        12156.666992          0.341301   0.144659
Model Summary:
 Number of Trees Number of Internal Trees Mod...
*/

// save model
import water.support.ModelSerializationSupport._
import java.net.URI
import _root_.hex.genmodel.{ModelMojoReader, MojoModel, MojoReaderBackendFactory}
import _root_.hex.Model
import water.persist.Persist
import water.{AutoBuffer, H2O, Key, Keyed}
import java.io._

// save model to POJO
def exportPOJOModel(model: Model[_, _, _], destination: URI): URI = {
    val destFile = new File(destination)
    val fos = new FileOutputStream(destFile)
    val writer = new model.JavaModelStreamWriter(false)
    try {
      writer.writeTo(fos)
    } finally {
      fos.close()
    }
    destination
}

exportPOJOModel(model, new File("/home/kt6238/myNFS/models/pojo/pojo_propensity_DRF_model_1514581721062_1.java").toURI)


// run new data through the model and store the predictioned ps with the data in a Hive table.

val pred = model.score(validation)('predict)

val predDF = asDataFrame(pred)(spark.sqlContext)

val f1 = validation.withColumn("id", monotonically_increasing_id())
val f2 = predDF.withColumn("id", monotonically_increasing_id())
val f3 = f2.join(f1, f2("id")===f1("id"), "outer").drop("id")

val f4 = f3.select("mob_ban","estimated_hh_income","household_size","number_adults","number_children","owner_renter","home_property_type","predict").
withColumnRenamed("predict","ps")

f4.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/u76/db/model_score_output").saveAsTable("u76.model_score_output")

// now this table is ready to append exposure status, then matching.
