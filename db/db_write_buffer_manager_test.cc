//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/db_test_util.h"
#include "db/write_thread.h"
#include "port/stack_trace.h"

namespace ROCKSDB_NAMESPACE {

class DBWriteBufferManagerTest : public DBTestBase,
                                 public testing::WithParamInterface<bool> {
 public:
  DBWriteBufferManagerTest()
      : DBTestBase("db_write_buffer_manager_test", /*env_do_fsync=*/false) {}
  bool cost_cache_;
};

TEST_P(DBWriteBufferManagerTest, SharedBufferAcrossCFs1) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }

  WriteOptions wo;
  wo.disableWAL = true;

  CreateAndReopenWithCF({"cf1", "cf2", "cf3"}, options);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  Flush(3);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(1), wo));
  Flush(0);

  // Write to "Default", "cf2" and "cf3".
  ASSERT_OK(Put(3, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(2, Key(1), DummyString(1), wo));

  ASSERT_OK(Put(3, Key(2), DummyString(40000), wo));
  // WriteBufferManager::buffer_size_ has exceeded after the previous write is
  // completed.

  // This make sures write will go through and if stall was in effect, it will
  // end.
  ASSERT_OK(Put(0, Key(2), DummyString(1), wo));
}

// Test Single DB with multiple writer threads get blocked when
// WriteBufferManager execeeds buffer_size_ and flush is waiting to be
// finished.
TEST_P(DBWriteBufferManagerTest, SharedWriteBufferAcrossCFs2) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }
  WriteOptions wo;
  wo.disableWAL = true;

  CreateAndReopenWithCF({"cf1", "cf2", "cf3"}, options);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  Flush(3);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(1), wo));
  Flush(0);

  // Write to "Default", "cf2" and "cf3". No flush will be triggered.
  ASSERT_OK(Put(3, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(2, Key(1), DummyString(1), wo));

  ASSERT_OK(Put(3, Key(2), DummyString(40000), wo));
  // WriteBufferManager::buffer_size_ has exceeded after the previous write is
  // completed.

  std::unordered_set<WriteThread::Writer*> w_set;
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  int num_writers = 4;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::atomic<int> thread_num(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        InstrumentedMutexLock lock(&mutex);
        wait_count_db++;
        cv.SignalAll();
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::WriteStall::Wait", [&](void* arg) {
        InstrumentedMutexLock lock(&mutex);
        WriteThread::Writer* w = reinterpret_cast<WriteThread::Writer*>(arg);
        w_set.insert(w);
        // Allow the flush to continue if all writer threads are blocked.
        if (w_set.size() == (unsigned long)num_writers) {
          TEST_SYNC_POINT(
              "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s = true;

  std::function<void(int)> writer = [&](int cf) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    Status tmp = Put(cf, Slice(key), DummyString(1), wo);
    InstrumentedMutexLock lock(&mutex);
    s = s && tmp.ok();
  };

  // Flow:
  // main_writer thread will write but will be blocked (as Flush will on hold,
  // buffer_size_ has exceeded, thus will create stall in effect).
  //  |
  //  |
  //  multiple writer threads will be created to write across multiple columns
  //  and they will be blocked.
  //  |
  //  |
  //  Last writer thread will write and when its blocked it will signal Flush to
  //  continue to clear the stall.

  threads.emplace_back(writer, 1);
  // Wait untill first thread (main_writer) writing to DB is blocked and then
  // create the multiple writers which will be blocked from getting added to the
  // queue because stall is in effect.
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }
  for (int i = 0; i < num_writers; i++) {
    threads.emplace_back(writer, i % 4);
  }
  for (auto& t : threads) {
    t.join();
  }

  ASSERT_TRUE(s);

  // Number of DBs blocked.
  ASSERT_EQ(wait_count_db, 1);
  // Number of Writer threads blocked.
  ASSERT_EQ(w_set.size(), num_writers);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Compared with `SharedWriteBufferAcrossCFs2` this test uses CF based write
// buffer manager CF level write buffer manager will not block write even
// exceeds the stall threshold DB level write buffer manager will block all
// write including CFs not use it.
TEST_P(DBWriteBufferManagerTest, SharedWriteBufferAcrossCFs3) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  std::shared_ptr<WriteBufferManager> cf_write_buffer_manager;
  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
    cf_write_buffer_manager.reset(new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
    cf_write_buffer_manager.reset(new WriteBufferManager(100000, nullptr, 1.0));
  }

  WriteOptions wo;
  wo.disableWAL = true;

  std::vector<std::string> cfs = {"cf1", "cf2", "cf3", "cf4", "cf5"};
  std::vector<std::shared_ptr<WriteBufferManager>> wbms = {
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      cf_write_buffer_manager,
      cf_write_buffer_manager};
  OpenWithCFWriteBufferManager(cfs, wbms, options);
  auto opts = db_->GetOptions();

  ASSERT_OK(Put(4, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(5, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(4, Key(1), DummyString(40000), wo));
  // Now, cf_write_buffer_manager reaches the stall level, but it will not block
  // the write

  int num_writers_total = 6;
  for (int i = 0; i < num_writers_total; i++) {
    ASSERT_OK(Put(i, Key(1), DummyString(1), wo));
  }

  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  Flush(3);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(1), wo));
  Flush(0);

  // Write to "Default", "cf2" and "cf3". No flush will be triggered.
  ASSERT_OK(Put(3, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(2, Key(1), DummyString(1), wo));

  ASSERT_OK(Put(3, Key(2), DummyString(40000), wo));
  // WriteBufferManager::buffer_size_ has exceeded after the previous write is
  // completed.

  std::unordered_set<WriteThread::Writer*> w_set;
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  int num_writers1 = 4;  // default, cf1-cf3
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::atomic<int> thread_num(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        InstrumentedMutexLock lock(&mutex);
        wait_count_db++;
        cv.SignalAll();
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::WriteStall::Wait", [&](void* arg) {
        InstrumentedMutexLock lock(&mutex);
        WriteThread::Writer* w = reinterpret_cast<WriteThread::Writer*>(arg);
        w_set.insert(w);
        // Allow the flush to continue if all writer threads are blocked.
        if (w_set.size() == (unsigned long)num_writers1) {
          TEST_SYNC_POINT(
              "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s = true;

  std::function<void(int)> writer = [&](int cf) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    Status tmp = Put(cf, Slice(key), DummyString(1), wo);
    InstrumentedMutexLock lock(&mutex);
    s = s && tmp.ok();
  };

  threads.emplace_back(writer, 1);
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }
  for (int i = 0; i < num_writers_total; i++) {
    threads.emplace_back(writer, i % 6);
  }
  for (auto& t : threads) {
    t.join();
  }

  ASSERT_TRUE(s);

  // Number of DBs blocked.
  ASSERT_EQ(wait_count_db, 1);
  // Number of Writer threads blocked.
  ASSERT_EQ(w_set.size(), num_writers_total);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Test multiple WriteBufferManager are independent to flush
TEST_P(DBWriteBufferManagerTest, SharedWriteBufferAcrossCFs4) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  std::shared_ptr<WriteBufferManager> cf_write_buffer_manager;
  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 0.0));
    cf_write_buffer_manager.reset(new WriteBufferManager(100000, cache, 0.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 0.0));
    cf_write_buffer_manager.reset(new WriteBufferManager(100000, nullptr, 0.0));
  }

  WriteOptions wo;
  wo.disableWAL = true;

  std::vector<std::string> cfs = {"cf1", "cf2", "cf3", "cf4", "cf5"};
  std::vector<std::shared_ptr<WriteBufferManager>> wbms = {
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      cf_write_buffer_manager,
      cf_write_buffer_manager};
  OpenWithCFWriteBufferManager(cfs, wbms, options);

  ASSERT_OK(Put(4, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(4, Key(1), DummyString(40000), wo));

  ASSERT_OK(Put(1, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(2, Key(1), DummyString(30000), wo));

  ASSERT_OK(Put(5, Key(1), DummyString(50000), wo));

  // The second WriteBufferManager::buffer_size_ has exceeded after the previous
  // write is completed.

  std::unordered_set<std::string> flush_cfs;
  std::vector<port::Thread> threads;
  int num_writers_total = 6;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::atomic<int> thread_num(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::Flush:ScheduleFlushReq", [&](void* arg) {
        InstrumentedMutexLock lock(&mutex);
        ColumnFamilyHandle* cfd = reinterpret_cast<ColumnFamilyHandle*>(arg);
        flush_cfs.insert(cfd->GetName());
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s = true;

  std::function<void(int, int)> writer = [&](int cf, int val_size) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    Status tmp = Put(cf, Slice(key), DummyString(val_size), wo);
    InstrumentedMutexLock lock(&mutex);
    s = s && tmp.ok();
  };

  for (int i = 0; i < num_writers_total; i++) {
    threads.emplace_back(writer, i % 6, 1);
  }
  for (auto& t : threads) {
    t.join();
  }
  threads.clear();

  ASSERT_TRUE(s);
  ASSERT_EQ(flush_cfs.size(), 1);
  ASSERT_NE(flush_cfs.find("cf4"), flush_cfs.end());
  flush_cfs.clear();

  ASSERT_OK(Put(0, Key(1), DummyString(30000), wo));

  for (int i = 0; i < num_writers_total; i++) {
    threads.emplace_back(writer, i % 6, 1);
  }
  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(flush_cfs.size(), 1);
  ASSERT_NE(flush_cfs.find("cf1"), flush_cfs.end());

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

TEST_P(DBWriteBufferManagerTest, FreeMemoryOnDestroy) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;   // this is never hit
  options.max_write_buffer_number = 5;  // Avoid unexpected stalling.
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }

  CreateAndReopenWithCF({"cf1", "cf2"}, options);
  std::string db2_name = test::PerThreadDBPath("free_memory_on_destroy_db2");
  DB* db2 = nullptr;
  ASSERT_OK(DestroyDB(db2_name, options));
  ASSERT_OK(DB::Open(options, db2_name, &db2));

  ASSERT_OK(db_->PauseBackgroundWork());
  ASSERT_OK(db2->PauseBackgroundWork());

  WriteOptions wo;
  wo.disableWAL = true;
  wo.no_slowdown = true;

  ASSERT_OK(db2->Put(wo, Key(1), DummyString(30000)));
  ASSERT_OK(Put(1, Key(1), DummyString(20000), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(40000), wo));

  // Decrease flush size, at least two cfs must be freed to not stall write.
  options.write_buffer_manager->SetFlushSize(50000);
  ASSERT_TRUE(Put(0, Key(1), DummyString(30000), wo).IsIncomplete());

  ASSERT_OK(db2->ContinueBackgroundWork());  // Close waits on pending jobs.
  // Thanks to `UnregisterDB`, we don't have to delete it to free up space.
  db2->Close();
  ASSERT_TRUE(Put(0, Key(1), DummyString(30000), wo).IsIncomplete());

  dbfull()->TEST_ClearBackgroundJobs();  // Jobs hold ref of cfd.
  ASSERT_OK(db_->DropColumnFamily(handles_[1]));
  ASSERT_TRUE(Put(0, Key(1), DummyString(30000), wo).IsIncomplete());
  ASSERT_OK(db_->DestroyColumnFamilyHandle(handles_[1]));
  handles_.erase(handles_.begin() + 1);
  ASSERT_OK(Put(0, Key(1), DummyString(30000), wo));

  delete db2;
  DestroyDB(db2_name, options);

  ASSERT_OK(db_->ContinueBackgroundWork());
}

TEST_P(DBWriteBufferManagerTest, DynamicFlushSize) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }

  CreateAndReopenWithCF({"cf1", "cf2"}, options);
  std::string db2_name = test::PerThreadDBPath("dynamic_flush_db2");
  DB* db2 = nullptr;
  ASSERT_OK(DestroyDB(db2_name, options));
  ASSERT_OK(DB::Open(options, db2_name, &db2));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  // Increase flush size can unblock writers.
  {
    WriteOptions wo;
    wo.disableWAL = true;
    ASSERT_OK(db2->Put(wo, Key(1), DummyString(60000)));
    ASSERT_OK(Put(1, Key(1), DummyString(30000), wo));
    ASSERT_OK(Put(0, Key(1), DummyString(30000), wo));
    // Write to DB.
    std::vector<port::Thread> threads;
    std::atomic<bool> ready{false};
    std::function<void(DB*)> write_db = [&](DB* db) {
      WriteOptions wopts;
      wopts.disableWAL = true;
      wopts.no_slowdown = true;
      ASSERT_TRUE(db->Put(wopts, Key(3), DummyString(1)).IsIncomplete());
      ready = true;
      wopts.no_slowdown = false;
      ASSERT_OK(db->Put(wopts, Key(3), DummyString(1)));
    };
    // Triggers db2 flush, but the flush is blocked.
    threads.emplace_back(write_db, db_);
    while (!ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Increase.
    options.write_buffer_manager->SetFlushSize(200000);
    for (auto& t : threads) {
      t.join();
    }
    TEST_SYNC_POINT("DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
  }
  // Decrease flush size triggers flush.
  {
    WriteOptions wo;
    wo.disableWAL = true;
    wo.no_slowdown = true;

    ASSERT_OK(Put(0, Key(1), DummyString(60000), wo));
    // All memtables must be flushed to satisfy the new flush_size.
    // Not too small because memtable has a minimum size.
    options.write_buffer_manager->SetFlushSize(10240);
    ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[0]));
    ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[1]));
    ASSERT_OK(db2->Put(wo, Key(1), DummyString(200000)));
  }

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();

  db2->Close();
  delete db2;
  DestroyDB(db2_name, options);
}

// Test multiple DBs get blocked when WriteBufferManager limit exceeds and flush
// is waiting to be finished but DBs tries to write meanwhile.
TEST_P(DBWriteBufferManagerTest, SharedWriteBufferLimitAcrossDB) {
  std::vector<std::string> dbnames;
  std::vector<DB*> dbs;
  int num_dbs = 3;

  for (int i = 0; i < num_dbs; i++) {
    dbs.push_back(nullptr);
    dbnames.push_back(
        test::PerThreadDBPath("db_shared_wb_db" + std::to_string(i)));
  }

  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }
  CreateAndReopenWithCF({"cf1", "cf2"}, options);

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(DestroyDB(dbnames[i], options));
    ASSERT_OK(DB::Open(options, dbnames[i], &(dbs[i])));
  }
  WriteOptions wo;
  wo.disableWAL = true;

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Put(wo, Key(1), DummyString(25000)));
  }
  // Insert to db_.
  ASSERT_OK(Put(0, Key(1), DummyString(25000), wo));

  // WriteBufferManager Limit exceeded.
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        {
          InstrumentedMutexLock lock(&mutex);
          wait_count_db++;
          cv.Signal();
          // Since this is the last DB, signal Flush to continue.
          if (wait_count_db == num_dbs + 1) {
            TEST_SYNC_POINT(
                "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
          }
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s = true;

  // Write to DB.
  std::function<void(DB*)> write_db = [&](DB* db) {
    Status tmp = db->Put(wo, Key(3), DummyString(1));
    InstrumentedMutexLock lock(&mutex);
    s = s && tmp.ok();
  };

  // Flow:
  // db_ will write and will be blocked (as Flush will on hold and will create
  // stall in effect).
  //  |
  //  multiple dbs writers will be created to write to that db and they will be
  //  blocked.
  //  |
  //  |
  //  Last writer will write and when its blocked it will signal Flush to
  //  continue to clear the stall.

  threads.emplace_back(write_db, db_);
  // Wait untill first DB is blocked and then create the multiple writers for
  // different DBs which will be blocked from getting added to the queue because
  // stall is in effect.
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }
  for (int i = 0; i < num_dbs; i++) {
    threads.emplace_back(write_db, dbs[i]);
  }
  for (auto& t : threads) {
    t.join();
  }

  ASSERT_TRUE(s);
  ASSERT_EQ(num_dbs + 1, wait_count_db);
  // Clean up DBs.
  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Close());
    ASSERT_OK(DestroyDB(dbnames[i], options));
    delete dbs[i];
  }

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Test multiple threads writing across multiple DBs and multiple columns get
// blocked when stall by WriteBufferManager is in effect.
TEST_P(DBWriteBufferManagerTest, SharedWriteBufferLimitAcrossDB1) {
  std::vector<std::string> dbnames;
  std::vector<DB*> dbs;
  int num_dbs = 3;

  for (int i = 0; i < num_dbs; i++) {
    dbs.push_back(nullptr);
    dbnames.push_back(
        test::PerThreadDBPath("db_shared_wb_db" + std::to_string(i)));
  }

  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }
  CreateAndReopenWithCF({"cf1", "cf2"}, options);

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(DestroyDB(dbnames[i], options));
    ASSERT_OK(DB::Open(options, dbnames[i], &(dbs[i])));
  }
  WriteOptions wo;
  wo.disableWAL = true;

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Put(wo, Key(1), DummyString(25000)));
  }
  // Insert to db_.
  ASSERT_OK(Put(0, Key(1), DummyString(25000), wo));

  // WriteBufferManager::buffer_size_ has exceeded after the previous write to
  // dbs[0] is completed.
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::unordered_set<WriteThread::Writer*> w_set;
  std::vector<port::Thread> writer_threads;
  std::atomic<int> thread_num(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        {
          InstrumentedMutexLock lock(&mutex);
          wait_count_db++;
          thread_num.fetch_add(1);
          cv.Signal();
          // Allow the flush to continue if all writer threads are blocked.
          if (thread_num.load(std::memory_order_relaxed) == 2 * num_dbs + 1) {
            TEST_SYNC_POINT(
                "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
          }
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::WriteStall::Wait", [&](void* arg) {
        WriteThread::Writer* w = reinterpret_cast<WriteThread::Writer*>(arg);
        {
          InstrumentedMutexLock lock(&mutex);
          w_set.insert(w);
          thread_num.fetch_add(1);
          // Allow the flush continue if all writer threads are blocked.
          if (thread_num.load(std::memory_order_relaxed) == 2 * num_dbs + 1) {
            TEST_SYNC_POINT(
                "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
          }
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s1 = true, s2 = true;
  // Write to multiple columns of db_.
  std::function<void(int)> write_cf = [&](int cf) {
    Status tmp = Put(cf, Key(3), DummyString(1), wo);
    InstrumentedMutexLock lock(&mutex);
    s1 = s1 && tmp.ok();
  };
  // Write to multiple DBs.
  std::function<void(DB*)> write_db = [&](DB* db) {
    Status tmp = db->Put(wo, Key(3), DummyString(1));
    InstrumentedMutexLock lock(&mutex);
    s2 = s2 && tmp.ok();
  };

  // Flow:
  // thread will write to db_ will be blocked (as Flush will on hold,
  // buffer_size_ has exceeded and will create stall in effect).
  //  |
  //  |
  //  multiple writers threads writing to different DBs and to db_ across
  //  multiple columns will be created and they will be blocked due to stall.
  //  |
  //  |
  //  Last writer thread will write and when its blocked it will signal Flush to
  //  continue to clear the stall.
  threads.emplace_back(write_db, db_);
  // Wait untill first thread is blocked and then create the multiple writer
  // threads.
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }

  for (int i = 0; i < num_dbs; i++) {
    // Write to multiple columns of db_.
    writer_threads.emplace_back(write_cf, i % 3);
    // Write to different dbs.
    threads.emplace_back(write_db, dbs[i]);
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto& t : writer_threads) {
    t.join();
  }

  ASSERT_TRUE(s1);
  ASSERT_TRUE(s2);

  // Number of DBs blocked.
  ASSERT_EQ(num_dbs + 1, wait_count_db);
  // Number of Writer threads blocked.
  ASSERT_EQ(w_set.size(), num_dbs);
  // Clean up DBs.
  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Close());
    ASSERT_OK(DestroyDB(dbnames[i], options));
    delete dbs[i];
  }

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Test multiple threads writing across multiple columns of db_ by passing
// different values to WriteOption.no_slown_down.
TEST_P(DBWriteBufferManagerTest, MixedSlowDownOptionsSingleDB) {
  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }
  WriteOptions wo;
  wo.disableWAL = true;

  CreateAndReopenWithCF({"cf1", "cf2", "cf3"}, options);

  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  Flush(3);
  ASSERT_OK(Put(3, Key(1), DummyString(1), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(1), wo));
  Flush(0);

  // Write to "Default", "cf2" and "cf3". No flush will be triggered.
  ASSERT_OK(Put(3, Key(1), DummyString(30000), wo));
  ASSERT_OK(Put(0, Key(1), DummyString(40000), wo));
  ASSERT_OK(Put(2, Key(1), DummyString(1), wo));
  ASSERT_OK(Put(3, Key(2), DummyString(40000), wo));

  // WriteBufferManager::buffer_size_ has exceeded after the previous write to
  // db_ is completed.

  std::unordered_set<WriteThread::Writer*> w_slowdown_set;
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  int num_writers = 4;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::atomic<int> thread_num(0);
  std::atomic<int> w_no_slowdown(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        {
          InstrumentedMutexLock lock(&mutex);
          wait_count_db++;
          cv.SignalAll();
        }
      });

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::WriteStall::Wait", [&](void* arg) {
        {
          InstrumentedMutexLock lock(&mutex);
          WriteThread::Writer* w = reinterpret_cast<WriteThread::Writer*>(arg);
          w_slowdown_set.insert(w);
          // Allow the flush continue if all writer threads are blocked.
          if (w_slowdown_set.size() + (unsigned long)w_no_slowdown.load(
                                          std::memory_order_relaxed) ==
              (unsigned long)num_writers) {
            TEST_SYNC_POINT(
                "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
          }
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s1 = true, s2 = true;

  std::function<void(int)> write_slow_down = [&](int cf) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions write_op;
    write_op.no_slowdown = false;
    Status tmp = Put(cf, Slice(key), DummyString(1), write_op);
    InstrumentedMutexLock lock(&mutex);
    s1 = s1 && tmp.ok();
  };

  std::function<void(int)> write_no_slow_down = [&](int cf) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions write_op;
    write_op.no_slowdown = true;
    Status tmp = Put(cf, Slice(key), DummyString(1), write_op);
    {
      InstrumentedMutexLock lock(&mutex);
      s2 = s2 && !tmp.ok();
      w_no_slowdown.fetch_add(1);
      // Allow the flush continue if all writer threads are blocked.
      if (w_slowdown_set.size() +
              (unsigned long)w_no_slowdown.load(std::memory_order_relaxed) ==
          (unsigned long)num_writers) {
        TEST_SYNC_POINT(
            "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
      }
    }
  };

  // Flow:
  // main_writer thread will write but will be blocked (as Flush will on hold,
  // buffer_size_ has exceeded, thus will create stall in effect).
  //  |
  //  |
  //  multiple writer threads will be created to write across multiple columns
  //  with different values of WriteOptions.no_slowdown. Some of them will
  //  be blocked and some of them will return with Incomplete status.
  //  |
  //  |
  //  Last writer thread will write and when its blocked/return it will signal
  //  Flush to continue to clear the stall.
  threads.emplace_back(write_slow_down, 1);
  // Wait untill first thread (main_writer) writing to DB is blocked and then
  // create the multiple writers which will be blocked from getting added to the
  // queue because stall is in effect.
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }

  for (int i = 0; i < num_writers; i += 2) {
    threads.emplace_back(write_no_slow_down, (i) % 4);
    threads.emplace_back(write_slow_down, (i + 1) % 4);
  }
  for (auto& t : threads) {
    t.join();
  }

  ASSERT_TRUE(s1);
  ASSERT_TRUE(s2);
  // Number of DBs blocked.
  ASSERT_EQ(wait_count_db, 1);
  // Number of Writer threads blocked.
  ASSERT_EQ(w_slowdown_set.size(), num_writers / 2);
  // Number of Writer threads with WriteOptions.no_slowdown = true.
  ASSERT_EQ(w_no_slowdown.load(std::memory_order_relaxed), num_writers / 2);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Test multiple threads writing across multiple columns of db_ and different
// dbs by passing different values to WriteOption.no_slown_down.
TEST_P(DBWriteBufferManagerTest, MixedSlowDownOptionsMultipleDB) {
  std::vector<std::string> dbnames;
  std::vector<DB*> dbs;
  int num_dbs = 4;

  for (int i = 0; i < num_dbs; i++) {
    dbs.push_back(nullptr);
    dbnames.push_back(
        test::PerThreadDBPath("db_shared_wb_db" + std::to_string(i)));
  }

  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;  // this is never hit
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 1.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 1.0));
  }
  CreateAndReopenWithCF({"cf1", "cf2"}, options);

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(DestroyDB(dbnames[i], options));
    ASSERT_OK(DB::Open(options, dbnames[i], &(dbs[i])));
  }
  WriteOptions wo;
  wo.disableWAL = true;

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Put(wo, Key(1), DummyString(25000)));
  }
  // Insert to db_.
  ASSERT_OK(Put(0, Key(1), DummyString(25000), wo));

  // WriteBufferManager::buffer_size_ has exceeded after the previous write to
  // dbs[0] is completed.
  std::vector<port::Thread> threads;
  int wait_count_db = 0;
  InstrumentedMutex mutex;
  InstrumentedCondVar cv(&mutex);
  std::unordered_set<WriteThread::Writer*> w_slowdown_set;
  std::vector<port::Thread> writer_threads;
  std::atomic<int> thread_num(0);
  std::atomic<int> w_no_slowdown(0);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0",
        "DBImpl::BackgroundCallFlush:start"}});

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        InstrumentedMutexLock lock(&mutex);
        wait_count_db++;
        cv.Signal();
        // Allow the flush continue if all writer threads are blocked.
        if (w_slowdown_set.size() +
                (unsigned long)(w_no_slowdown.load(std::memory_order_relaxed) +
                                wait_count_db) ==
            (unsigned long)(2 * num_dbs + 1)) {
          TEST_SYNC_POINT(
              "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
        }
      });

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::WriteStall::Wait", [&](void* arg) {
        WriteThread::Writer* w = reinterpret_cast<WriteThread::Writer*>(arg);
        InstrumentedMutexLock lock(&mutex);
        w_slowdown_set.insert(w);
        // Allow the flush continue if all writer threads are blocked.
        if (w_slowdown_set.size() +
                (unsigned long)(w_no_slowdown.load(std::memory_order_relaxed) +
                                wait_count_db) ==
            (unsigned long)(2 * num_dbs + 1)) {
          TEST_SYNC_POINT(
              "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  bool s1 = true, s2 = true;
  std::function<void(DB*)> write_slow_down = [&](DB* db) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions write_op;
    write_op.no_slowdown = false;
    Status tmp = db->Put(write_op, Slice(key), DummyString(1));
    InstrumentedMutexLock lock(&mutex);
    s1 = s1 && tmp.ok();
  };

  std::function<void(DB*)> write_no_slow_down = [&](DB* db) {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions write_op;
    write_op.no_slowdown = true;
    Status tmp = db->Put(write_op, Slice(key), DummyString(1));
    {
      InstrumentedMutexLock lock(&mutex);
      s2 = s2 && !tmp.ok();
      w_no_slowdown.fetch_add(1);
      if (w_slowdown_set.size() +
              (unsigned long)(w_no_slowdown.load(std::memory_order_relaxed) +
                              wait_count_db) ==
          (unsigned long)(2 * num_dbs + 1)) {
        TEST_SYNC_POINT(
            "DBWriteBufferManagerTest::SharedWriteBufferAcrossCFs:0");
      }
    }
  };

  // Flow:
  // first thread will write but will be blocked (as Flush will on hold,
  // buffer_size_ has exceeded, thus will create stall in effect).
  //  |
  //  |
  //  multiple writer threads will be created to write across multiple columns
  //  of db_ and different DBs with different values of
  //  WriteOptions.no_slowdown. Some of them will be blocked and some of them
  //  will return with Incomplete status.
  //  |
  //  |
  //  Last writer thread will write and when its blocked/return it will signal
  //  Flush to continue to clear the stall.
  threads.emplace_back(write_slow_down, db_);
  // Wait untill first thread writing to DB is blocked and then
  // create the multiple writers.
  {
    InstrumentedMutexLock lock(&mutex);
    while (wait_count_db != 1) {
      cv.Wait();
    }
  }

  for (int i = 0; i < num_dbs; i += 2) {
    // Write to multiple columns of db_.
    writer_threads.emplace_back(write_slow_down, db_);
    writer_threads.emplace_back(write_no_slow_down, db_);
    // Write to different DBs.
    threads.emplace_back(write_slow_down, dbs[i]);
    threads.emplace_back(write_no_slow_down, dbs[i + 1]);
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto& t : writer_threads) {
    t.join();
  }

  ASSERT_TRUE(s1);
  ASSERT_TRUE(s2);
  // Number of DBs blocked.
  ASSERT_EQ((num_dbs / 2) + 1, wait_count_db);
  // Number of writer threads writing to db_ blocked from getting added to the
  // queue.
  ASSERT_EQ(w_slowdown_set.size(), num_dbs / 2);
  // Number of threads with WriteOptions.no_slowdown = true.
  ASSERT_EQ(w_no_slowdown.load(std::memory_order_relaxed), num_dbs);

  // Clean up DBs.
  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Close());
    ASSERT_OK(DestroyDB(dbnames[i], options));
    delete dbs[i];
  }

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// Test write can progress even if manual compaction and background work is
// paused.
TEST_P(DBWriteBufferManagerTest, BackgroundWorkPaused) {
  std::vector<std::string> dbnames;
  std::vector<DB*> dbs;
  int num_dbs = 4;

  for (int i = 0; i < num_dbs; i++) {
    dbs.push_back(nullptr);
    dbnames.push_back(
        test::PerThreadDBPath("db_shared_wb_db" + std::to_string(i)));
  }

  Options options = CurrentOptions();
  options.arena_block_size = 4096;
  options.write_buffer_size = 500000;          // this is never hit
  options.avoid_flush_during_shutdown = true;  // avoid blocking destroy forever
  std::shared_ptr<Cache> cache = NewLRUCache(4 * 1024 * 1024, 2);
  ASSERT_LT(cache->GetUsage(), 256 * 1024);
  cost_cache_ = GetParam();

  // Do not enable write stall.
  if (cost_cache_) {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, cache, 0.0));
  } else {
    options.write_buffer_manager.reset(
        new WriteBufferManager(100000, nullptr, 0.0));
  }
  DestroyAndReopen(options);

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(DestroyDB(dbnames[i], options));
    ASSERT_OK(DB::Open(options, dbnames[i], &(dbs[i])));
  }

  dbfull()->DisableManualCompaction();
  ASSERT_OK(dbfull()->PauseBackgroundWork());
  for (int i = 0; i < num_dbs; i++) {
    dbs[i]->DisableManualCompaction();
    ASSERT_OK(dbs[i]->PauseBackgroundWork());
  }

  WriteOptions wo;
  wo.disableWAL = true;

  // Arrange the score like this: (this)2000, (0-th)100000, (1-th)1, ...
  ASSERT_OK(Put(Key(1), DummyString(2000), wo));
  for (int i = 1; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Put(wo, Key(1), DummyString(1)));
  }
  // Exceed the limit.
  ASSERT_OK(dbs[0]->Put(wo, Key(1), DummyString(100000)));
  // Write another one to trigger the flush.
  ASSERT_OK(Put(Key(3), DummyString(1), wo));

  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->ContinueBackgroundWork());
    ASSERT_OK(
        static_cast_with_check<DBImpl>(dbs[i])->TEST_WaitForFlushMemTable());
    std::string property;
    EXPECT_TRUE(dbs[i]->GetProperty("rocksdb.num-files-at-level0", &property));
    int num = atoi(property.c_str());
    ASSERT_EQ(num, 0);
  }
  ASSERT_OK(dbfull()->ContinueBackgroundWork());
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  std::string property;
  EXPECT_TRUE(dbfull()->GetProperty("rocksdb.num-files-at-level0", &property));
  int num = atoi(property.c_str());
  ASSERT_EQ(num, 1);

  // Clean up DBs.
  for (int i = 0; i < num_dbs; i++) {
    ASSERT_OK(dbs[i]->Close());
    ASSERT_OK(DestroyDB(dbnames[i], options));
    delete dbs[i];
  }
}

INSTANTIATE_TEST_CASE_P(DBWriteBufferManagerTest, DBWriteBufferManagerTest,
                        testing::Bool());

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
