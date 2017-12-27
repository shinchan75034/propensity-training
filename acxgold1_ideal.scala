import org.apache.spark.sql.functions.approxCountDistinct
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions
spark.conf.set("spark.sql.broadcastTimeout", 3600)

val dtv = spark.sqlContext.sql("select mob_ban, estimated_hh_income, household_size, number_adults, number_children, owner_renter,home_property_type from u76.dtv_universe_acx")
val uvs = spark.sqlContext.sql("select mob_ban, estimated_hh_income, household_size, number_adults, number_children, owner_renter,home_property_type from u76.uverse_universe_acx")

val selHH = dtv.unionAll(uvs)

selHH.registerTempTable("tbl")

val selTbl = spark.sqlContext.sql("select * from tbl where estimated_hh_income is not null and household_size is not null and number_adults is not null and number_children is not null and owner_renter is not null and home_property_type is not null")
// verified each column has legit values.
// val sel = selHH.filter($"household_size".isNotNull).filter($"household_size".isNotNull).filter($"number_adults".isNotNull).filter($"number_children".isNotNull)

selTbl.distinct.write.mode("overwrite").option("path", "/apps/advertising/audience_measurement/kt6238/u76/db/acx_gold_1").saveAsTable("kt6.acx_gold_2")
