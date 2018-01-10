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


val resp = spark.sqlContext.sql("select * from u76.common_response_with_duration")

// select visit between 30 and 240 minutes
val sel_resp = resp.filter($"visit_duration_minutes" >= 30 && $"visit_duration_minutes" <= 240)

val resp_hh = sel_resp.select("mob_ban").withColumnRenamed("mob_ban", "wban").distinct
val acx = spark.sqlContext.sql("select mob_ban, cast(estimated_hh_income as int), cast(household_size as int), cast(number_adults as int), cast(number_children as int), cast(owner_renter as string), cast(home_property_type as string), tvonly, webonly, tvandweb, exposurestatus as exposed from kt6.acx_gold_2_vs")

val j1 = acx.join(resp_hh, acx("mob_ban") === resp_hh("wban"), "left_outer").
withColumn("response", when($"wban".isNull, 0).otherwise(1)).drop("wban")

val newDF = j1.randomSplit(Array(0.6, 0.2, 0.2))
val (train, test, validation) = (newDF(0), newDF(1), newDF(2))
//
train.coalesce(1).write.format("csv").option("header", "true").save("/apps/advertising/audience_measurement/kt6238/u76/hdfs/acx_gold_exposure_response_train.csv")
test.coalesce(1).write.format("csv").option("header", "true").save("/apps/advertising/audience_measurement/kt6238/u76/hdfs/acx_gold_exposure_response_test.csv")
validation.coalesce(1).write.format("csv").option("header", "true").save("/apps/advertising/audience_measurement/kt6238/u76/hdfs/acx_gold_exposure_response_validation.csv")
