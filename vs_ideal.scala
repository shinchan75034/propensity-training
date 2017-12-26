import org.apache.spark.sql.functions.approxCountDistinct
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions
import org.apache.spark.sql.hive.HiveContext


val fileName = "/apps/advertising/audience_measurement/u76/mutualexclusive/impr/hh-impr/tvonly/part*"
val loc = "hdfs:///".concat(fileName)
val datRdd = sc.textFile(loc).map(line => line.split("\\|"))
case class Dat(wban: String = "", DMA: String = "", First_exposed_ts: String = "", last_exposed_ts: String = "")
val datDF = datRdd.map(f => Dat(f(0).toString, f(1).toString, f(2).toString, f(3).toString)).toDF().select("wban").distinct
val tvHH = datDF.withColumn("tvOnly", lit(1)) //  2703012


val fileName = "/apps/advertising/audience_measurement/u76/mutualexclusive/impr/hh-impr/webonly/part*"
val loc = "hdfs:///".concat(fileName)
val datRdd = sc.textFile(loc).map(line => line.split("\\|"))

val datDF =  datRdd.map(f => Dat(f(0).toString, f(1).toString, f(2).toString, f(3).toString)).toDF().select("wban").distinct
val webHH = datDF.withColumn("webOnly", lit(1)) // 113234

val fileName = "/apps/advertising/audience_measurement/u76/mutualexclusive/impr/hh-impr/tvandweb/part*"
val loc = "hdfs:///".concat(fileName)
val datRdd = sc.textFile(loc).map(line => line.split("\\|"))
val datDF =  datRdd.map(f => Dat(f(0).toString, f(1).toString, f(2).toString, f(3).toString)).toDF().select("wban").distinct
val tvandwebHH = datDF.withColumn("tvandweb", lit(1)) //157021

val universe = spark.sqlContext.sql("select * from u76.acx_gold_1")

val j1 = universe.join(broadcast(tvHH), universe("mob_ban")===tvHH("wban"), "left_outer").drop(tvHH("wban"))
val j2 = j1.join(broadcast(webHH), j1("mob_ban")===webHH("wban"), "left_outer").drop(webHH("wban"))
val j3 = j2.join(broadcast(tvandwebHH), j2("mob_ban")===tvandwebHH("wban"), "left_outer").drop(tvandwebHH("wban"))

val newDF = j3.na.fill(0, Seq("tvOnly", "webOnly", "tvandweb"))
val newDF2 = newDF.withColumn("exposureStatus", when($"tvOnly"===0 and $"webOnly"===0 and $"tvandweb"===0, 0).otherwise(1))

val newDF3 = newDF2.filter("estimated_hh_income is not null")
newDF3.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/kt6238/u76/db/acx_gold_2_vs").saveAsTable("kt6.acx_gold_2_vs")


val fileLocation="/apps/advertising/audience_measurement/kt6238/u76/hdfs/acx_gold_2_vs"
newDF3.write.format("csv").option("header","true").save(fileLocation)
