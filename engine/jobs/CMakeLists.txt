nova_add_module(jobs
    SOURCES
        src/job_system.cpp
    PUBLIC_DEPS
        Nova::core)

find_package(Threads REQUIRED)
target_link_libraries(nova_jobs PRIVATE Threads::Threads)
