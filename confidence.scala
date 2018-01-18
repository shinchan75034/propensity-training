import org.apache.spark.sql.functions.approxCountDistinct
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions
import org.apache.spark.sql.hive.HiveContext
// implicit val sqlContext = spark.sqlContext
//val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
hiveContext.setConf("hive.exec.dynamic.partition", "true")
hiveContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")
// val fileName = "/apps/advertising/audience_measurement/u71/selected_audience/lift_analysis/clf/target_network/2016-08-01_2016-09-11/monte_carlo/unique_lift/results/part*"

val fileName = "/apps/advertising/audience_measurement/u71/selected_audience/lift_analysis/clf/target_network/2016-08-01_2016-09-11/monte_carlo_5000_10/unique_lift/results/part*"
val loc = "hdfs:///".concat(fileName)
val datRdd = sc.textFile(loc).map(line => line.split("\\|"))
// datRdd.filter(f => f(0) == null | f(1) == null | f(2) == null | f(3) == null).count // check data quality
// datRdd.filter(f => f.length != 4).count / /check for incorrect delimiter.
case class Dat(itr_idx: String = "", exposed_response_rate: Double = 0.0, nonexposed_response_rate: Double = 0.0, lift: Double = 0.0)
val datDF = datRdd.map(f => Dat(f(0).toString, f(1).toDouble, f(2).toDouble, f(3).toDouble)).toDF()

datDF.createOrReplaceTempView("datTbl")

val stats = spark.sqlContext.sql("select min(lift) as minimum, percentile(lift, 0.25) as q1, percentile(lift, 0.5) as q2, percentile(lift, 0.75) as q3, max(lift) as maximum from datTbl ")
val stats95 = spark.sqlContext.sql("select percentile(lift, 0.025) as lo, avg(lift) as average, percentile(lift, 0.975) as hi from datTbl ")
val stats90 = spark.sqlContext.sql("select percentile(lift, 0.05) as lo, avg(lift) as average, percentile(lift, 0.95) as hi from datTbl ")
val mx = spark.sqlContext.sql("select max(cast(itr_idx as bigint)seel) from datTbl")

/*
2000, 5
+-------------------+-------------------+--------------------+
|                 lo|            average|                  hi|
+-------------------+-------------------+--------------------+
|0.03504444025565452|0.04820406428458464|0.061293303117100496|
+-------------------+-------------------+--------------------+
2000, 10
+-------------------+-------------------+--------------------+
|                 lo|            average|                  hi|
+-------------------+-------------------+--------------------+
|0.03929124837264232|0.04806224491256184|0.057181519447945674|
+-------------------+-------------------+--------------------+
5000, 5
+-------------------+-------------------+-------------------+
|                 lo|            average|                 hi|
+-------------------+-------------------+-------------------+
|0.03456802751403911|0.04790192862451772|0.06085173667999793|
+-------------------+-------------------+-------------------+
5000, 10
+-------------------+-------------------+-------------------+
|                 lo|            average|                 hi|
+-------------------+-------------------+-------------------+
|0.03868895071103009|0.04778788722735784|0.05689187301624715|
+-------------------+-------------------+-------------------+


*/
