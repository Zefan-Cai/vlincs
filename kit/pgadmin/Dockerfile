# pgAdmin with the VLINCS gallery connection + password baked in at the perms pgAdmin and libpq require.
# Why a build (not a bind-mount): libpq IGNORES a pgpass file unless it is mode 0600 AND owned by the
# running user (pgadmin, uid 5050). A host bind-mount keeps the host's uid/0644, so libpq rejects it and
# the pre-seeded password "fails". COPY + chown/chmod here gives the file the exact owner+mode libpq wants.
ARG PGADMIN_TAG=9
FROM dpage/pgadmin4:${PGADMIN_TAG}
USER root
COPY servers.json /pgadmin4/servers.json
COPY pgpass /pgadmin4/pgpass
RUN chown pgadmin:root /pgadmin4/servers.json /pgadmin4/pgpass \
 && chmod 0644 /pgadmin4/servers.json \
 && chmod 0600 /pgadmin4/pgpass
USER pgadmin
