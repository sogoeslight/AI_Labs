package com.sogoeslight.task.preparation

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object Converter {

  def catToBin(ds: Array[Array[String]], categoricalColumns: List[Int]): Array[Array[String]] = {

    @deprecated("Multimap", "2.13.0")
    def oneCatToBin(ds: ArrayBuffer[ArrayBuffer[String]], header: ArrayBuffer[String], categoricalColumnIndex: Int):
    ArrayBuffer[ArrayBuffer[String]] = {
      var binaryData: ArrayBuffer[ArrayBuffer[String]] = ArrayBuffer[ArrayBuffer[String]]()

      // creating data structure for storing all possible categorical values
      val categoricalValues = new mutable.HashMap[String, mutable.Set[Int]] with mutable.MultiMap[String, Int]
      categoricalValues.clear()

      for ((row, index) <- ds.zipWithIndex) { // loop over dataset
        // add all categorical values into MultiMap[String, Set[Int]
        if (!categoricalValues.entryExists(row(categoricalColumnIndex), _ == index))
          categoricalValues.addBinding(row(categoricalColumnIndex), index)
      }

      for ((row, index) <- ds.zipWithIndex) { // adding data to the new fully-binary dataset
        binaryData += row
        binaryData(index).remove(categoricalColumnIndex)
      }

      header.remove(categoricalColumnIndex) // removing outdated attribute
      for (k <- categoricalValues) {
        header += k._1 // adding new ones
        for ((row, index) <- binaryData.zipWithIndex) { // adding binary values
          if (k._2.contains(index)) row += "1" else row += "0"
        }
      }

      binaryData.prepend(header) // adding attributes to the dataset
    }

    categoricalColumns match {
      case Nil => Data.cleanDataset
      case columns =>
        val header: ArrayBuffer[String] = ds(0).to(ArrayBuffer)
        var binaryData: ArrayBuffer[ArrayBuffer[String]] = ds.tail.to(ArrayBuffer).map(_.to(ArrayBuffer))
        for ((col, index) <- columns.zipWithIndex) {
          binaryData = oneCatToBin(binaryData, header, col - index)
        }
        binaryData.toArray.map(_.toArray)
    }
  }
}
